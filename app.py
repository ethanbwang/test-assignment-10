import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(page_title="My AI Chat", layout="wide")

hf_token = st.secrets.get("HF_TOKEN", None)
CHATS_DIR = Path("chats")
MEMORY_PATH = Path("memory.json")
DEFAULT_TITLE_PREFIX = "New Chat"
CHAT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def current_timestamp():
    return datetime.now(timezone.utc).isoformat()


def chat_path(chat_id):
    return CHATS_DIR / f"{chat_id}.json"


def create_chat(title=DEFAULT_TITLE_PREFIX):
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "created_at": current_timestamp(),
        "messages": [],
    }


def chat_has_messages(chat):
    return bool(chat.get("messages"))


def save_chat(chat):
    if not chat_has_messages(chat):
        return
    CHATS_DIR.mkdir(exist_ok=True)
    chat_path(chat["id"]).write_text(json.dumps(chat, indent=2), encoding="utf-8")


def load_chats():
    CHATS_DIR.mkdir(exist_ok=True)
    chats = []

    for path in sorted(CHATS_DIR.glob("*.json")):
        try:
            chat = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(chat, dict):
            continue

        chat_id = chat.get("id")
        title = chat.get("title")
        created_at = chat.get("created_at")
        messages = chat.get("messages")

        if not isinstance(chat_id, str) or not chat_id:
            continue
        if not isinstance(title, str) or not title:
            title = DEFAULT_TITLE_PREFIX
        if not isinstance(created_at, str) or not created_at:
            created_at = current_timestamp()
        if not isinstance(messages, list):
            messages = []

        chats.append(
            {
                "id": chat_id,
                "title": title,
                "created_at": created_at,
                "messages": messages,
            }
        )

    chats.sort(key=lambda chat: chat["created_at"], reverse=True)
    return chats


def ensure_memory_file():
    if not MEMORY_PATH.exists():
        MEMORY_PATH.write_text("{}\n", encoding="utf-8")


def load_memory():
    ensure_memory_file()
    try:
        memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        memory = {}

    if not isinstance(memory, dict):
        memory = {}

    return memory


def save_memory(memory):
    ensure_memory_file()
    MEMORY_PATH.write_text(json.dumps(memory, indent=2), encoding="utf-8")


def clear_memory():
    st.session_state.memory = {}
    save_memory(st.session_state.memory)


def merge_memory(existing, new_data):
    for key, value in new_data.items():
        if value in (None, "", [], {}):
            continue

        if key not in existing:
            existing[key] = value
            continue

        current = existing[key]

        if isinstance(current, dict) and isinstance(value, dict):
            merge_memory(current, value)
            continue

        current_values = current if isinstance(current, list) else [current]
        new_values = value if isinstance(value, list) else [value]

        merged_values = []
        for item in current_values + new_values:
            if item in (None, "", [], {}):
                continue
            if item not in merged_values:
                merged_values.append(item)

        if len(merged_values) == 1:
            existing[key] = merged_values[0]
        else:
            existing[key] = merged_values


def build_memory_system_prompt(memory):
    if not memory:
        return None

    return {
        "role": "system",
        "content": (
            "Use the following saved user memory as background context for personalization. "
            "Do not claim it explicitly unless relevant. "
            f"User memory: {json.dumps(memory, ensure_ascii=True)}"
        ),
    }


def build_chat_messages(messages, memory):
    request_messages = []
    memory_prompt = build_memory_system_prompt(memory)
    if memory_prompt is not None:
        request_messages.append(memory_prompt)
    request_messages.extend(messages)
    return request_messages


def get_active_chat():
    active_chat_id = st.session_state.get("active_chat_id")
    for chat in st.session_state.get("chats", []):
        if chat["id"] == active_chat_id:
            return chat
    return None


def ensure_chat_state():
    if "chats" not in st.session_state:
        st.session_state.chats = load_chats()

    if not st.session_state.chats:
        st.session_state.chats = [create_chat()]

    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = st.session_state.chats[0]["id"]

    if get_active_chat() is None:
        replacement_chat = create_chat()
        st.session_state.chats.insert(0, replacement_chat)
        st.session_state.active_chat_id = replacement_chat["id"]


def start_new_chat():
    new_chat = create_chat()
    st.session_state.chats.insert(0, new_chat)
    st.session_state.active_chat_id = new_chat["id"]


def delete_chat(chat_id):
    was_active = chat_id == st.session_state.active_chat_id
    st.session_state.chats = [chat for chat in st.session_state.chats if chat["id"] != chat_id]

    try:
        chat_path(chat_id).unlink()
    except FileNotFoundError:
        pass

    if not st.session_state.chats or was_active:
        replacement_chat = create_chat()
        st.session_state.chats.insert(0, replacement_chat)
        st.session_state.active_chat_id = replacement_chat["id"]
    elif st.session_state.active_chat_id == chat_id:
        st.session_state.active_chat_id = st.session_state.chats[0]["id"]


def extract_stream_content(data):
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    choice = choices[0]
    if not isinstance(choice, dict):
        return None

    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            )

    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content

    text = choice.get("text")
    if isinstance(text, str):
        return text

    return None


def request_hugging_face_reply(messages):
    response = requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers={"Authorization": f"Bearer {hf_token}"},
        json={
            "model": CHAT_MODEL,
            "messages": messages,
            "max_tokens": 512,
            "stream": True,
        },
        timeout=30,
        stream=True,
    )
    response.raise_for_status()

    def chunk_generator():
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            line = raw_line.strip()
            if not line.startswith("data:"):
                continue

            payload = line[5:].strip()
            if payload == "[DONE]":
                break

            try:
                data = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid streaming payload: {payload}") from exc

            content = extract_stream_content(data)
            if content:
                time.sleep(0.03)
                yield content

    return response, chunk_generator()


def request_memory_update(user_message):
    extraction_messages = [
        {
            "role": "system",
            "content": (
                "Extract durable user memory from the user's message. "
                "Return only a JSON object. "
                "Include traits such as name, preferred_language, interests, communication_style, "
                "favorite_topics, or other stable personal preferences when explicitly stated. "
                "Do not infer unsupported facts. "
                "Return {} if there is nothing worth storing."
            ),
        },
        {
            "role": "user",
            "content": f"User message: {user_message}",
        },
    ]

    response = requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers={"Authorization": f"Bearer {hf_token}"},
        json={
            "model": CHAT_MODEL,
            "messages": extraction_messages,
            "max_tokens": 256,
            "stream": False,
            "response_format": {"type": "json_object"},
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    memory_update = json.loads(content)
    if not isinstance(memory_update, dict):
        raise ValueError("Memory extraction did not return a JSON object.")
    return memory_update


def summarize_title(text, max_length=32):
    compact_text = " ".join(text.split())
    if len(compact_text) <= max_length:
        return compact_text
    return f"{compact_text[: max_length - 3].rstrip()}..."


st.title("My AI Chat")
st.markdown(
    """
    <style>
    .stMainBlockContainer {
        padding-bottom: 6rem;
    }

    [data-testid="stChatInput"] {
        position: sticky;
        bottom: 0;
        background: var(--background-color);
        padding-top: 0.75rem;
        padding-bottom: 0.75rem;
        z-index: 100;
    }

    [data-testid="stSidebar"] .stButton:first-of-type {
        position: sticky;
        top: 0;
        z-index: 100;
        background: var(--background-color);
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }

    [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        gap: 0.35rem;
    }

    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }

    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child button {
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.15s ease-in-out;
    }

    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:hover > div[data-testid="column"]:last-child button {
        opacity: 1;
        pointer-events: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ensure_memory_file()

if hf_token is None:
    st.error("HF_TOKEN is not set. Add it to `.streamlit/secrets.toml` before using the app.")
    st.stop()

ensure_chat_state()

if "memory" not in st.session_state:
    st.session_state.memory = load_memory()

with st.sidebar:
    if st.button("New Chat", use_container_width=True, key="new_chat_button"):
        start_new_chat()
        st.rerun()

    with st.expander("User Memory", expanded=False):
        st.json(st.session_state.memory)
        if st.button("Clear memory", use_container_width=True, key="clear_memory_button"):
            clear_memory()
            st.rerun()

    for chat in st.session_state.chats:
        chat_button_col, delete_button_col = st.columns([5, 1])

        with chat_button_col:
            if st.button(
                chat["title"],
                key=f"open_chat_{chat['id']}",
                use_container_width=True,
                type="primary" if chat["id"] == st.session_state.active_chat_id else "secondary",
                help=f"Created {chat['created_at']}",
            ):
                st.session_state.active_chat_id = chat["id"]
                st.rerun()

        with delete_button_col:
            if st.button("x", key=f"delete_chat_{chat['id']}", use_container_width=True):
                delete_chat(chat["id"])
                st.rerun()

active_chat = get_active_chat()

for message in active_chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Send a message")

if prompt:
    user_message = {"role": "user", "content": prompt}
    active_chat["messages"].append(user_message)

    if active_chat["title"].startswith(DEFAULT_TITLE_PREFIX):
        active_chat["title"] = summarize_title(prompt)

    save_chat(active_chat)

    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):
            response, stream = request_hugging_face_reply(
                build_chat_messages(active_chat["messages"], st.session_state.memory)
            )
            with response:
                reply = st.write_stream(stream)

        active_chat["messages"].append({"role": "assistant", "content": reply})
        save_chat(active_chat)
    except requests.exceptions.RequestException as exc:
        error_details = ""
        if getattr(exc, "response", None) is not None:
            try:
                error_details = exc.response.text
            except Exception:
                error_details = ""
        st.error(f"Hugging Face API request failed: {exc}")
        if error_details:
            st.error(error_details)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        st.error(f"Received an unexpected response from the Hugging Face API: {exc}")
    else:
        try:
            memory_update = request_memory_update(prompt)
            if memory_update:
                merge_memory(st.session_state.memory, memory_update)
                save_memory(st.session_state.memory)
        except requests.exceptions.RequestException as exc:
            st.warning(f"Memory extraction request failed: {exc}")
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            st.warning(f"Memory extraction returned an unexpected response: {exc}")
