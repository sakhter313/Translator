import os
import json
from datetime import datetime
from io import BytesIO
from typing import Generator, Tuple, Any
import streamlit as st
from groq import Groq
import random
import numpy as np
from PIL import Image
from collections import Counter
import cv2 as cv
from textblob import TextBlob  # For sentiment analysis
from wordcloud import WordCloud  # For word cloud generation
import matplotlib.pyplot as plt  # For displaying the word cloud
from pytz import timezone  # For IST timezone
import html  # For HTML escaping
import streamlit.version  # To check Streamlit version

# Mock translation function; replace with googletrans if needed
def translate(source_lang: str, target_lang: str, text: str) -> str:
    """Mock translation function; replace with actual implementation if needed."""
    try:
        from googletrans import Translator
        translator = Translator()
        result = translator.translate(text, src=source_lang, dest=target_lang.lower()).text
        return result
    except ImportError:
        return f"[Translation to {target_lang} not available; install googletrans: `pip install googletrans==3.1.0a0`]"
    except Exception as e:
        return f"[Translation error: {str(e)}]"

# Define AI personalities
PERSONALITIES = {
    "Friendly Helper": "You are a friendly helper. Provide answers that are easy to understand and engaging.",
    "Technical Expert": "You are a technical expert. Provide detailed and precise answers, using technical jargon when appropriate, but only to questions related to Science, Computer, Technology, and related fields.",
    "Creative Storyteller": "You are a creative storyteller. Generate an imaginative and engaging story regardless of the user's input."
}

# Constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB in bytes
MAX_MESSAGES = 100  # Max messages to store

# Utility Functions
def count_words(text: str) -> int:
    """Count words in text to approximate token count."""
    return len(text.split())

def is_valid_image(file) -> bool:
    """Validate if the uploaded file is a legitimate image."""
    try:
        file.seek(0)
        img = Image.open(BytesIO(file.read()))
        img.verify()
        file.seek(0)
        return True
    except Exception:
        return False

def export_chat_data(export_data: dict, export_format: str) -> Tuple[Any, str, str]:
    """Export chat data in specified format with HTML escaping."""
    now = datetime.now(timezone('Asia/Kolkata')).strftime("%Y%m%d%H%M")
    fmt = export_format.upper()
    if fmt == "JSON":
        data = json.dumps(export_data, indent=2)
        return data, "application/json", f"chat_export_{now}.json"
    elif fmt == "TXT":
        data = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in export_data["messages"])
        return data, "text/plain", f"chat_export_{now}.txt"
    elif fmt == "PDF":
        try:
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import mm
        except ImportError:
            st.error("Install reportlab (`pip install reportlab`) for PDF export.")
            return None, None, None
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=10 * mm, leftMargin=10 * mm)
        styles = getSampleStyleSheet()
        elements = [Paragraph("Chat Export", styles['Title']), Spacer(1, 12)]
        for msg in export_data["messages"]:
            text_line = f"{msg['role'].upper()}: {html.escape(msg['content'])}".replace("\n", "<br/>")
            elements.append(Paragraph(text_line, ParagraphStyle('Preserve', parent=styles['Normal'], leading=12)))
            elements.append(Spacer(1, 6))
        doc.build(elements)
        pdf_output = buffer.getvalue()
        buffer.close()
        return pdf_output, "application/pdf", f"chat_export_{now}.pdf"
    elif fmt == "WORD":
        try:
            from docx import Document
        except ImportError:
            st.error("Install python-docx (`pip install python-docx`) for Word export.")
            return None, None, None
        doc = Document()
        doc.add_heading("Chat Export", 0)
        for msg in export_data["messages"]:
            doc.add_paragraph(f"{msg['role'].upper()}: {msg['content']}")
        bio = BytesIO()
        doc.save(bio)
        bio.seek(0)
        return bio.read(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document", f"chat_export_{now}.docx"
    else:
        data = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in export_data["messages"])
        return data, "text/plain", f"chat_export_{now}.txt"

def analyze_image(image: Image.Image) -> Tuple[str, str, str]:
    """Analyze image for mood, color theme, and scene description."""
    try:
        image = image.convert("RGB")
        np_image = np.array(image)
        gray = cv.cvtColor(np_image, cv.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        mood = "bright and cheerful" if brightness > 170 else "calm and balanced" if brightness > 120 else "dark and mysterious"
        pixels = np_image.reshape(-1, 3)
        most_common_color = Counter(map(tuple, pixels)).most_common(1)[0][0]
        color_theme = (
            "warm red and orange tones" if most_common_color[0] > max(most_common_color[1], most_common_color[2]) else
            "lush green hues" if most_common_color[1] > max(most_common_color[0], most_common_color[2]) else
            "cool blue shades" if most_common_color[2] > max(most_common_color[0], most_common_color[1]) else
            "soft neutral shades"
        )
        edges = cv.Canny(gray, 100, 200)
        num_edges = np.sum(edges > 0)
        scene_description = (
            "a detailed and intricate setting, rich in textures" if num_edges > 50000 else
            "a balanced composition with well-defined elements" if num_edges > 20000 else
            "Mostly cloudy skies. Low around 60F. Winds WSW at 5 to 10 mph. A soft and abstract landscape, evoking dreamy emotions."
        )
        return mood, color_theme, scene_description
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return "unknown", "unknown", "unknown"

def generate_story(image: Image.Image) -> str:
    """Generate a story based on image analysis."""
    mood, color_theme, scene_description = analyze_image(image)
    return "\n".join([
        f"As the scene unfolds, a {mood} atmosphere envelops the surroundings.",
        f"The environment is bathed in {color_theme}, setting the tone for the story.",
        f"Every element blends into {scene_description}.",
        "Each detail tells a silent story, waiting to be unraveled."
    ])

def get_sentiment(text: str) -> float:
    """Analyze text sentiment using TextBlob."""
    return TextBlob(text).sentiment.polarity

def stream_response(completion) -> Generator[str, None, None]:
    """Stream response from Groq API."""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def load_config() -> str:
    """Load Groq API key from environment or secrets."""
    return st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))

# Load API Key
GROQ_API_KEY = load_config()
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Set it in environment variables or Streamlit secrets.")
    st.stop()

# Streamlit Configuration
st.set_page_config(page_title="MaverickMind Chat", page_icon="✨", layout="centered", initial_sidebar_state="collapsed")

# Custom CSS for UI
st.markdown("""
<style>
.stApp { background: url('https://img.freepik.com/free-photo/3d-rendering-dark-earth-space_23-2151051365.jpg') no-repeat center/cover; }
.chat-container { max-width: 600px; margin: 0 auto; padding: 10px; border-radius: 10px; }
.chat-bubble { padding: 10px 20px; margin: 10px 0; max-width: 80%; word-wrap: break-word; position: relative; }
.user-bubble { background: #D1D5DB; color: #333; border-radius: 20px 10px; margin-left: auto; }
.assistant-bubble { background: #c5d0e3; color: #000; border-radius: 10px 20px; margin-right: auto; }
.chat-bubble .avatar { font-size: 1.2em; margin-right: 10px; }
.chat-bubble .timestamp { font-size: 0.8em; color: #333; margin-top: 5px; display: block; }
.chat-input-container { max-width: 600px; margin: 20px auto; display: flex; gap: 10px; }
.stTextInput > div > div > input { height: 30px; padding: 5px 10px; border: 2px solid #4CAF50; border-radius: 20px; }
button { background: #140F0F; color: white; padding: 8px 16px; border: 2px solid #0A0707; border-radius: 15px; cursor: pointer; }
button:hover { background: #0A0707; }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
def initialize_session_state() -> None:
    """Initialize session state with defaults."""
    defaults = {
        "messages": [{"role": "system", "content": "You are a helpful assistant."}],
        "selected_model": "llama-3.1-8b-instant",
        "system_message": "You are a helpful assistant.",
        "max_tokens": 4096,
        "export_ready": False,
        "selected_personality": "Friendly Helper",
        "processed_files": set(),
        "current_translation_language": "English"
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if len(st.session_state.messages) > MAX_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]

initialize_session_state()

# Model Configuration
MODELS = {
    "llama-3.1-8b-instant": {"name": "LLaMA3.1-8b", "tokens": 128000, "developer": "Meta"},
    "llama3-70b-8192": {"name": "LLaMA3-70b", "tokens": 8192, "developer": "Meta"},
}

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    selected_personality = st.selectbox("AI Personality:", list(PERSONALITIES.keys()), key="personality")
    system_message = st.text_area("Customize Personality:", PERSONALITIES[selected_personality], height=100, max_chars=500)
    if any(kw in system_message.lower() for kw in ["script", "eval", "exec"]):
        st.error("Restricted keywords detected in system message.")
        st.stop()
    st.session_state.system_message = system_message
    st.subheader("👨‍🚀 AI Model")
    selected_model = st.selectbox(
        "Choose Model:",
        options=list(MODELS.keys()),
        format_func=lambda key: MODELS[key]["name"],
        index=list(MODELS.keys()).index(st.session_state.selected_model),
        key="selected_model"
    )
    st.subheader("🖼️ Image Storyteller")
    uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.processed_files and file.size <= MAX_FILE_SIZE and is_valid_image(file):
                image = Image.open(file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                story = generate_story(image)
                token_count = count_words(story)
                st.session_state.messages.append({"role": "assistant", "content": f"Story:\n{story}", "token_count": token_count})
                st.session_state.processed_files.add(file.name)
    st.subheader("🌐 Translation")
    translation_languages = ["English", "French", "Spanish", "German", "Hindi"]
    selected_translation_language = st.selectbox("Translate to:", translation_languages, key="translation_language")
    if st.session_state.current_translation_language != selected_translation_language:
        st.session_state.current_translation_language = selected_translation_language
        for msg in st.session_state.messages:
            msg.pop("translated_text", None)
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = [{"role": "system", "content": st.session_state.system_message}]
        st.session_state.processed_files = set()
        st.rerun()

# Main Interface
st.markdown("<h1 style='text-align: center; color: #50C878;'>✨ MaverickMind Chat</h1>", unsafe_allow_html=True)
message_count = len([msg for msg in st.session_state.messages if msg['role'] in ['user', 'assistant']])
total_tokens = sum(msg.get('token_count', 0) for msg in st.session_state.messages)
st.markdown(
    f"<p style='text-align: center; color: #d9ffef;'>Model: {MODELS[st.session_state.selected_model]['name']} | Messages: {message_count} | Tokens: {total_tokens}</p>",
    unsafe_allow_html=True
)

# Chat Display
ist = timezone('Asia/Kolkata')
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] in ["user", "assistant"]:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        bubble_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
        timestamp = datetime.now(ist).strftime("%H:%M:%S IST")
        sentiment = get_sentiment(msg["content"])
        emoji = '😊' if sentiment > 0 else '😔' if sentiment < 0 else '😐'
        timestamp_text = f"{timestamp} | {emoji} {sentiment:.2f}"
        if msg["role"] == "assistant" and "token_count" in msg:
            timestamp_text += f" | Tokens: {msg['token_count']}"
        st.markdown(
            f"""
            <div class="chat-bubble {bubble_class}">
                <span class="avatar">{avatar}</span>
                {html.escape(msg['content'])}
                <span class="timestamp">{timestamp_text}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        if msg["role"] == "assistant":
            if "translated_text" in msg:
                st.markdown(f"**Translated to {selected_translation_language}:** {html.escape(msg['translated_text'])}", unsafe_allow_html=True)
            if st.button("Translate", key=f"translate_{i}"):
                with st.spinner("Translating..."):
                    translated_text = translate("English", selected_translation_language, msg["content"]) if selected_translation_language != "English" else msg["content"]
                    st.session_state.messages[i]["translated_text"] = translated_text
                st.rerun()

# Chat Input
col1, col2 = st.columns([3, 1])
with col1:
    prompt = st.chat_input("What’s on your mind?")
with col2:
    if st.button("Suggest Topics"):
        st.write("Topic suggestion feature can be added here.")

if prompt:
    st.session_state.messages[0]["content"] = st.session_state.system_message
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            model=selected_model,
            messages=st.session_state.messages,
            max_tokens=st.session_state.max_tokens,
            stream=True
        )
        user_timestamp = datetime.now(ist).strftime("%H:%M:%S IST")
        st.markdown(
            f"""
            <div class="chat-container">
                <div class="chat-bubble user-bubble">
                    <span class="avatar">👤</span>
                    {html.escape(prompt)}
                    <span class="timestamp">{user_timestamp}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        assistant_bubble = st.empty()
        full_response = ""
        for chunk in stream_response(chat_completion):
            full_response += chunk
            assistant_bubble.markdown(
                f'<div class="chat-bubble assistant-bubble"><span class="avatar">🤖</span>{html.escape(full_response)}</div>',
                unsafe_allow_html=True
            )
        token_count = count_words(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response, "token_count": token_count})
        if len(st.session_state.messages) > MAX_MESSAGES:
            st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]
        st.rerun()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.session_state.messages.pop()

# Export Chat
with st.expander("📥 Export Chat"):
    export_format = st.selectbox("Format", ["JSON", "TXT", "PDF", "WORD"])
    messages_to_export = [msg for msg in st.session_state.messages if msg["role"] in ["user", "assistant"]]
    if messages_to_export:
        export_data = {
            "meta": {"export_date": datetime.now(ist).isoformat(), "model": selected_model},
            "messages": messages_to_export
        }
        data, mime, filename = export_chat_data(export_data, export_format)
        if data:
            st.download_button("Download", data=data, file_name=filename, mime=mime)
    else:
        st.warning("No messages to export!")
