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

# Ensure Streamlit version is at least 1.11.1 to avoid known vulnerabilities
if not hasattr(streamlit, '__version__') or streamlit.__version__ < '1.11.1':
    st.error("Streamlit version must be >= 1.11.1 to avoid security vulnerabilities. Please update Streamlit.")
    st.stop()

# Define AI personalities with updated instructions
PERSONALITIES = {
    "Friendly Helper": "You are a friendly helper. Provide answers that are easy to understand and engaging.",
    "Technical Expert": "You are a technical expert. Provide detailed and precise answers, using technical jargon when appropriate, but only to questions related to Science, Computer, Technology, and related fields. If the user asks about unrelated topics, politely ask them to inquire about Science, Computer, Technology, etc.",
    "Creative Storyteller": "You are a creative storyteller. Regardless of the user's input, generate an imaginative and engaging story. You can use the user's input as inspiration for the story if relevant."
}

# Maximum file size for uploads (5MB)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB in bytes

# Maximum number of messages to store in session state
MAX_MESSAGES = 100

# Function to count words (approximating token count)
def count_words(text: str) -> int:
    """Count the number of words in a text string to approximate token count."""
    return len(text.split())

def is_valid_image(file) -> bool:
    """Validate that the uploaded file is a legitimate image."""
    try:
        file.seek(0)
        img = Image.open(BytesIO(file.read()))
        img.verify()  # Verify that it is an image
        file.seek(0)  # Reset file pointer
        return True
    except Exception:
        return False

def export_chat_data(export_data: dict, export_format: str) -> Tuple[Any, str, str]:
    """Export chat data in specified format with HTML escaping for safety."""
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
        style_title = styles['Title']
        style_preserve = ParagraphStyle('Preserve', parent=styles['Normal'], leading=12)
        elements = [Paragraph("Chat Export", style_title), Spacer(1, 12)]
        
        for msg in export_data["messages"]:
            text_line = f"{msg['role'].upper()}: {html.escape(msg['content'])}".replace("\n", "<br/>")
            elements.append(Paragraph(text_line, style_preserve))
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
        
        from docx import Document

doc = Document()
msg = {"role": "assistant", "tokens": 4096, "developer": "OpenAI", "color": "#00FF00"}

# Corrected f-string
p = doc.add_paragraph(f"{msg['role'].upper()}: tokens={msg.get('tokens', 8192)}, developer='{msg.get('developer', 'Meta')}', color='{msg.get('color', '#1877F2')}'")
doc.save("output.docx")

# Translation function updated to use deep-translator
def translate(source_lang: str, target_lang: str, text: str) -> str:
    """Translate text to the specified language using deep-translator."""
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source=source_lang, target=target_lang.lower())
        translated = translator.translate(text)
        return translated
    except ImportError:
        return "[Translation not available; ensure deep-translator is installed]"
    except Exception as e:
        return f"[Translation error: {str(e)}]"

def analyze_image(image: Image.Image) -> Tuple[str, str, str]:
    """Analyze image for mood, color theme, and scene description with error handling."""
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
            "a soft and abstract landscape, evoking dreamy emotions"
        )
        return mood, color_theme, scene_description
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return "unknown", "unknown", "unknown"

def generate_story(image: Image.Image) -> str:
    """Generate a story based on image analysis."""
    mood, color_theme, scene_description = analyze_image(image)
    opening_lines = [
        f"As the scene unfolds, a {mood} atmosphere envelops the surroundings.",
        f"The world captured here radiates a {mood} essence, drawing the observer in."
    ]
    color_lines = [
        f"The environment is bathed in {color_theme}, setting the tone for the unfolding story.",
        f"Hues of {color_theme} paint a mesmerizing backdrop, adding depth to the scene."
    ]
    detail_lines = [
        f"Every element blends harmoniously, creating {scene_description}.",
        f"The image's textures weave together {scene_description}, evoking deep emotions."
    ]
    conclusion_lines = [
        "It is a tale of fleeting moments, captured in the delicate balance of time.",
        "Each detail tells a silent story, waiting to be unraveled by the keen observer."
    ]
    return "\n".join([
        random.choice(opening_lines),
        random.choice(color_lines),
        random.choice(detail_lines),
        random.choice(conclusion_lines)
    ])

def get_sentiment(text: str) -> float:
    """Analyze sentiment of text using TextBlob."""
    blob = TextBlob(text)
    return blob.sentiment.polarity

def stream_response(completion) -> Generator[str, None, None]:
    """Stream response from Groq API."""
    full_response = []
    for chunk in completion:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response.append(content)
            yield content

def load_config() -> str:
    """Load Groq API key securely from environment variables or Streamlit secrets."""
    return st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))

GROQ_API_KEY = load_config()
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please set it in environment variables or Streamlit secrets.")
    st.stop()

# Streamlit Configuration with Enhanced CSS
st.set_page_config(page_title="MaverickMind Chat", page_icon="✨", layout="centered", initial_sidebar_state="collapsed")

# Enhanced CSS for a polished chat UI
st.markdown("""
<style>
.stApp {
    background-image: url('https://img.freepik.com/free-photo/colorful-abstract-nebula-space-background_53876-111355.jpg');
    background-size: cover !important;
}
.chat-container {
    max-width: 600px;
    margin: 0 auto;
    padding: 10px;
    border-radius: 10px;
}
.chat-bubble {
    padding: 10px 20px;
    margin: 30px 0;
    max-width: 80%;
    word-wrap: break-word;
    position: relative;
}
.user-bubble {
    background-color: #D1D5DB;
    color: #333;
    border-radius: 20px 10px;
    align-self: center;
    margin-left: auto;
}
.assistant-bubble {
    background-color: #c5d0e3;
    color: #000f0b;
    border-radius: 10px 20px;
    align-self: flex-start;
    margin-right: auto;
}
.chat-bubble .avatar {
    font-size: 1.2em;
    margin-right: 10px;
    display: inline-block;
    vertical-align: middle;
}
.chat-bubble .timestamp {
    font-size: 0.8em;
    color: #333;
    margin-top: 5px;
    display: block;
}
.chat-input-container {
    max-width: 600px;
    margin: 20px auto;
    display: flex;
    align-items: center;
    gap: 10px;
}
.stTextInput > div > div > input {
    height: 30px;
    padding: 5px 10px;
    font-size: 14px;
    border: 2px solid #4CAF50;
    border-radius: 20px;
}
button {
    background-color: #140F0F;
    color: #261712;
    padding: 8px 16px;
    border: 2px solid #0A0707;
    border-radius: 15px;
    cursor: pointer;
    transition: background-color 0.3s;
}
button:hover {
    background-color: #0A0707;
}
.suggestions-container {
    max-height: 150px;
    color: #5e2606;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 5px;
    background-color: #f9f9f9;
    margin-top: 10px;
}
.stExpander {
    border: 2px solid #edf5f2;
    border-radius: 5px;
}
.stExpander p {
    color: #05fa11;
}
</style>
""", unsafe_allow_html=True)

# Session State Management
def initialize_session_state() -> None:
    """Initialize session state with defaults and manage message limits."""
    defaults = {
        "messages": [{
            "role": "system",
            "content": "You are a helpful assistant. Answer queries concisely and ask clarifying questions if needed."
        }],
        "selected_model": "llama-3.1-8b-instant",
        "system_message": "You are a helpful assistant. Answer queries concisely and ask clarifying questions if needed.",
        "max_tokens": 4096,
        "export_ready": False,
        "selected_personality": "Friendly Helper",
        "processed_files": set()
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if len(st.session_state.messages) > MAX_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]

initialize_session_state()

# Sidebar Configuration
with st.sidebar:
    st.markdown("<h2 style='color: #4CAF50;'>⚙️ Settings</h2>", unsafe_allow_html=True)
    selected_personality = st.selectbox("Choose AI Personality:", list(PERSONALITIES.keys()), key="personality")
    system_message = st.text_area(
        "Customize Personality:",
        value=PERSONALITIES[selected_personality],
        height=100,
        max_chars=500
    )
    if any(keyword in system_message.lower() for keyword in ["script", "eval", "exec"]):
        st.error("System message contains restricted keywords. Please revise.")
        st.stop()
    st.session_state.system_message = system_message
    st.divider()
    st.subheader("👨‍🚀 AI Model")
    selected_model = st.selectbox(
        "Choose Your AI Companion:",
        options=list(MODELS.keys()),
        format_func=lambda key: MODELS[key]["name"],
        index=list(MODELS.keys()).index(st.session_state.selected_model),
        key="selected_model"
    )
    st.divider()
    st.subheader("🖼️ Image Storyteller")
    uploaded_files = st.file_uploader(
        "Upload Images",
        type=["png", "jpg", "jpeg", "jfif"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files:
                if uploaded_file.size > MAX_FILE_SIZE:
                    st.error(f"File {uploaded_file.name} exceeds 5MB limit.")
                    continue
                if not is_valid_image(uploaded_file):
                    st.error(f"File {uploaded_file.name} is not a valid image.")
                    continue
                image = Image.open(uploaded_file)
                st.image(image, caption="Your Image", use_container_width=True)
                story = generate_story(image)
                token_count = count_words(story)
                st.session_state.messages.append({"role": "assistant", "content": f"Story:\n{story}", "token_count": token_count})
                st.session_state.processed_files.add(uploaded_file.name)
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = [{"role": "system", "content": st.session_state.system_message}]
        st.session_state.export_ready = False
        st.session_state.processed_files = set()
        st.rerun()

# Main Interface
st.markdown("<h1 style='text-align: center; color: #50C878;'>✨ MaverickMind Chat</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #fcfcfc;'>Your personal AI companion for conversation and creativity.</p>",
    unsafe_allow_html=True
)

# Display selected model and message count
message_count = len([msg for msg in st.session_state.messages if msg['role'] in ['user', 'assistant']])
selected_model_name = MODELS[st.session_state.selected_model]["name"]
st.markdown(
    f"<p style='text-align: center; color: #d9ffef;'>Selected Model: {selected_model_name} | Message Count: {message_count}</p>",
    unsafe_allow_html=True
)

# Chat Display
ist = timezone('Asia/Kolkata')
for message in st.session_state.messages:
    if message["role"] in ["user", "assistant"]:
        avatar = "👤" if message["role"] == "user" else "🤖"
        bubble_class = "user-bubble" if message["role"] == "user" else "assistant-bubble"
        timestamp = datetime.now(ist).strftime("%H:%M:%S IST")
        sentiment = get_sentiment(message["content"])
        emoji = '😊' if sentiment > 0 else '😔' if sentiment < 0 else '😐'
        timestamp_text = f"{timestamp} | {emoji} {sentiment:.2f}"
        if message["role"] == "assistant" and "token_count" in message:
            timestamp_text += f" | Tokens: {message['token_count']}"
        st.markdown(
            f"""
            <div class="chat-bubble {bubble_class}">
                <span class="avatar">{avatar}</span>
                {html.escape(message['content'])}
                <span class="timestamp">{timestamp_text}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# Response Actions
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    last_idx = len(st.session_state.messages) - 1
    btn_cols = st.columns(3)
    if btn_cols[0].button("👍 Like", key=f"good_{last_idx}"):
        st.success("Thanks for the feedback!")
    if btn_cols[1].button("👎 Dislike", key=f"bad_{last_idx}"):
        st.error("Noted. I’ll try to do better!")
    if btn_cols[2].button("✍🏻 Edit", key=f"edit_{last_idx}"):
        edited_text = st.text_area(
            "Edit response:",
            value=st.session_state.messages[-1]["content"],
            key=f"edit_area_{last_idx}"
        )
        if st.button("Save Edit", key=f"save_edit_{last_idx}"):
            st.session_state.messages[-1]["content"] = edited_text
            st.session_state.messages[-1]["token_count"] = count_words(edited_text)
            st.rerun()

# Chat Input
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div style='margin-bottom: 40px;'><br><br></div>", unsafe_allow_html=True)
col1, col2 = st.columns([3, 1])
with col1:
    prompt = st.chat_input("What’s on your mind?", key="chat_input")
with col2:
    if st.button("Suggest Topics", key="suggest_topics"):
        with st.spinner("Generating suggestions..."):
            conversation_history = "\n".join(
                f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages
                if msg['role'] in ['user', 'assistant']
            )
            prompt_text = (
                f"Based on the following conversation, suggest three related topics or questions that the user might be interested in:\n\n{conversation_history}\n\n"
            )
            try:
                client = Groq(api_key=GROQ_API_KEY)
                completion = client.chat.completions.create(
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt_text}]
                )
                suggestions = completion.choices[0].message.content
                st.markdown(
                    f"""
                    <div class="suggestions-container">
                        <h3 style="margin: 0; font-size: 16px;">Suggested Topics/Questions</h3>
                        <p>{html.escape(suggestions)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error generating suggestions: {str(e)}")

if prompt:
    st.session_state.messages[0]["content"] = st.session_state.system_message
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        client = Groq(api_key=GROQ_API_KEY)
        # Filter messages to include only "role" and "content"
        api_messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]
        chat_completion = client.chat.completions.create(
            model=selected_model,
            messages=api_messages,
            max_tokens=st.session_state.max_tokens,
            stream=True
        )
        # Display user message immediately
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
        # Initialize assistant's bubble
        assistant_bubble = st.empty()
        assistant_bubble.markdown(
            '<div class="chat-bubble assistant-bubble"><span class="avatar">🤖</span>Generating response...</div>',
            unsafe_allow_html=True
        )
        # Stream the response
        full_response = ""
        for chunk in stream_response(chat_completion):
            full_response += chunk
            assistant_bubble.markdown(
                '<div class="chat-bubble assistant-bubble"><span class="avatar">🤖</span>' + html.escape(full_response) + '</div>',
                unsafe_allow_html=True
            )
            import time
            time.sleep(0.1)  # Optional delay for smooth streaming
        # Calculate token count and append message with token_count for UI
        token_count = count_words(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response, "token_count": token_count})
        # Limit messages
        if len(st.session_state.messages) > MAX_MESSAGES:
            st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]
        st.rerun()
    except Exception as e:
        st.error(f"⚠️ Oops! Something went wrong: {str(e)}")
        st.session_state.messages.pop()

# Additional Features: Word Cloud and Export
st.divider()
col1, col2 = st.columns(2)
with col1:
    with st.expander("🌥️ Word Cloud"):
        if st.button("Generate Word Cloud"):
            all_text = ' '.join(
                [msg['content'] for msg in st.session_state.messages if msg['role'] in ['user', 'assistant']]
            )
            if all_text.strip():
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
            else:
                st.write("No conversation to visualize yet!")
with col2:
    with st.expander("📥 Export Chat"):
        export_full = st.checkbox("Export full conversation", value=False)
        export_format = st.selectbox("Format", ["JSON", "TXT", "PDF", "WORD"], label_visibility="collapsed")
        messages_to_export = (
            st.session_state.messages if export_full else
            [msg for msg in st.session_state.messages if msg["role"] == "assistant"]
        )
        if messages_to_export:
            export_data = {
                "meta": {
                    "export_date": datetime.now(ist).isoformat(),
                    "model": selected_model,
                    "system_message": st.session_state.system_message
                },
                "messages": messages_to_export
            }
            data, mime, filename = export_chat_data(export_data, export_format)
            if data:
                st.download_button("Download Chat", data=data, file_name=filename, mime=mime)
        else:
            st.warning("Nothing to export yet!")
