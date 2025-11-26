import streamlit as st
from openai import OpenAI
import os

# ---------------------- CONFIG ----------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("Hugging Face token not found! Set HF_TOKEN in environment variables.")
    st.stop()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

st.set_page_config(
    page_title="Health Assistant 🩺",
    page_icon="🩺",
    layout="wide",
)

# ---------------------- DARK THEME CSS ----------------------
st.markdown("""
    <style>
        body {
            background-color: #1E1E1E;
            color: #E0E0E0;
        }
        .chat-user {
            background-color: #0F4C75;
            color: white;
            padding: 10px;
            border-radius: 15px;
            margin: 5px 0;
            max-width: 80%;
            text-align: right;
        }
        .chat-assistant {
            background-color: #3282B8;
            color: white;
            padding: 10px;
            border-radius: 15px;
            margin: 5px 0;
            max-width: 80%;
            text-align: left;
        }
        .chat-container {
            max-height: 70vh;
            overflow-y: auto;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- SYSTEM PROMPT ----------------------
SYSTEM_PROMPT = """
You are a friendly, safe health assistant.
Rules:
- Provide general health information only.
- Do NOT diagnose.
- Do NOT provide medication doses.
- Do NOT provide emergency instructions.
- Encourage consulting a doctor for serious concerns.
- Keep answers simple and easy to understand.
"""

# ---------------------- SAFETY FILTER ----------------------
def is_unsafe(query: str) -> bool:
    unsafe_keywords = [
        "suicide", "kill myself", "overdose", "bleeding", "heart attack",
        "stroke", "choking", "unconscious", "not breathing",
        "mg", "dosage", "dose",
        "do i have", "diagnose", "is this cancer"
    ]
    return any(word in query.lower() for word in unsafe_keywords)

# ---------------------- GENERATE REPLY ----------------------
def generate_reply(user_input: str) -> str:
    if is_unsafe(user_input):
        return ("⚠️ This seems like a medically sensitive question.\n"
                "Please consult a doctor or emergency service for safe help.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    completion = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct-0905",
        messages=messages,
        temperature=0.9,
        max_tokens=300
    )

    return completion.choices[0].message.content

# ---------------------- UI ----------------------
st.markdown("<h1 style='text-align:center;'>🩺 Health Assistant</h1>", unsafe_allow_html=True)
st.write("Welcome to your Personal Health Assistant. Ask freely about your health")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Button to clear chat
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.info("Chat cleared! Start a new conversation.")

# Scrollable chat container
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            st.markdown(f"<div class='chat-user'>👤 {content}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-assistant'>🤖 {content}</div>", unsafe_allow_html=True)

# User input
user_input = st.chat_input("Ask a health question...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Placeholder for assistant message
    assistant_placeholder = chat_container.empty()
    with assistant_placeholder.container():
        st.markdown("<div class='chat-assistant'>🤖 Typing...</div>", unsafe_allow_html=True)

    # Generate assistant reply
    reply = generate_reply(user_input)
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # Update placeholder with actual reply
    assistant_placeholder.markdown(f"<div class='chat-assistant'>🤖 {reply}</div>", unsafe_allow_html=True)
