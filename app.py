import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import google.generativeai as genai
from dotenv import load_dotenv
import pickle
from langdetect import detect
from datetime import datetime

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(page_title="RAG Bible Chat", page_icon="📖", layout="wide")

# -------------------------------
# Session state initialization
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = None
if "indexes" not in st.session_state:
    st.session_state.indexes = {}
if "conversations" not in st.session_state:
    st.session_state.conversations = {"Conversation 1": []}
if "current_conv" not in st.session_state:
    st.session_state.current_conv = "Conversation 1"

# -------------------------------
# Gemini API
# -------------------------------
load_dotenv("api.env")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# -------------------------------
# Load SentenceTransformer model
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Build or load FAISS indexes
# -------------------------------
@st.cache_resource
def build_or_load_indexes(text):
    model = load_model()

    def chunk_text(size, overlap):
        chunks, start = [], 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start += size - overlap
        return chunks

    indexes = {}
    for name, size, overlap in [("small", 500, 200), ("large", 2000, 400)]:
        index_file, chunk_file = f"bible_{name}.index", f"chunks_{name}.pkl"

        if os.path.exists(index_file) and os.path.exists(chunk_file):
            index = faiss.read_index(index_file)
            with open(chunk_file, "rb") as f:
                chunks = pickle.load(f)
        else:
            chunks = chunk_text(size, overlap)
            embeddings = model.encode(chunks, show_progress_bar=True)
            embeddings = np.array(embeddings, dtype="float32")
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            faiss.write_index(index, index_file)
            with open(chunk_file, "wb") as f:
                pickle.dump(chunks, f)

        indexes[name] = (index, chunks)

    return model, indexes

# -------------------------------
# Retrieve relevant chunks
# -------------------------------
def retrieve(query, top_k=5):
    if any(ch in query for ch in [":", "mstari", "verse"]):
        index, chunks = st.session_state.indexes["small"]
    else:
        index, chunks = st.session_state.indexes["large"]

    model = load_model()
    query_emb = model.encode([query])
    query_emb = np.array(query_emb, dtype="float32")
    distances, indices = index.search(query_emb, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

# -------------------------------
# Build prompt with language detection
# -------------------------------
def build_prompt(user_query, context):
    try:
        lang = detect(user_query)
    except:
        lang = "en"
    instruction = "Answer in clear Swahili." if lang == "sw" else "Answer in clear English."

    history = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]]
    )

    return f"""
You are a Bible assistant. {instruction}
Always include book, chapter, and verse when quoting.
If context is partial, still try to answer naturally.

Conversation so far:
{history}

Bible context:
{context}

User question:
{user_query}

Answer:
"""

# -------------------------------
# Chat bubble renderer
# -------------------------------
def chat_bubble(text, role="assistant"):
    if role == "user":
        st.markdown(f"""
        <div style="
            max-width: 70%;
            margin-left: auto;
            background: #DCF8C6;
            padding: 12px 14px;
            border-radius: 12px;
            margin-bottom: 8px;
            text-align: right;">
            🙋‍♂️ {text}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            max-width: 70%;
            margin-right: auto;
            background: #E8E8E8;
            padding: 12px 14px;
            border-radius: 12px;
            margin-bottom: 8px;
            text-align: left;
            white-space: pre-line;">
            🤖 {text}
        </div>""", unsafe_allow_html=True)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
.chat-container {
    max-height: 65vh;
    overflow-y: auto;
    padding: 10px;
    background-color: #f9f9f9;
}
.input-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: white;
    padding: 10px;
    border-top: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar: History & Controls
# -------------------------------
with st.sidebar:
    st.header("📖 Bible Document")
    # Add new conversation
    if st.button("➕ New Conversation"):
        new_name = f"Conversation {len(st.session_state.conversations)+1}"
        st.session_state.conversations[new_name] = []
        st.session_state.current_conv = new_name
        st.session_state.messages = []
    # Show conversation history
    st.subheader("Conversations")
    conv_names = list(st.session_state.conversations.keys())
    sel_conv = st.selectbox("Select Conversation", conv_names, index=conv_names.index(st.session_state.current_conv))
    if sel_conv != st.session_state.current_conv:
        st.session_state.current_conv = sel_conv
        st.session_state.messages = st.session_state.conversations[sel_conv]
    if st.button("🧹 Clear Current Chat"):
        st.session_state.messages = []
        st.session_state.conversations[st.session_state.current_conv] = []

# -------------------------------
# Load Bible & indexes
# -------------------------------
if not st.session_state.indexes:
    try:
        with open("bible.txt", "r", encoding="utf-8") as f:
            text = f.read()
        st.session_state.model, st.session_state.indexes = build_or_load_indexes(text)
        st.success("✅ Bible loaded and indexed! Start chatting.")
    except FileNotFoundError:
        st.error("⚠️ bible.txt not found! Place it in the same folder as app.py.")

# -------------------------------
# Page title & Chat window
# -------------------------------
st.title("📖 READ THE BIBLE (RAG Chatbot)")
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
if not st.session_state.messages:
    chat_bubble("👋 Hello! I am ready to help you with the Bible.", "assistant")
for msg in st.session_state.messages:
    chat_bubble(msg["content"], role=msg["role"])
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Fixed input
# -------------------------------
disabled = not st.session_state.indexes
st.markdown('<div class="input-container">', unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    user_query = st.text_input("✍️ Write your question here...", disabled=disabled)
    sent = st.form_submit_button("Send", disabled=disabled)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Handle user input
# -------------------------------
if sent and user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    retrieved_texts = retrieve(user_query, top_k=5)
    context = "\n\n".join(retrieved_texts)
    qa_prompt = build_prompt(user_query, context)
    try:
        response = gemini_model.generate_content(qa_prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"⚠️ Sorry, an error occurred: {e}"
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Save conversation
    st.session_state.conversations[st.session_state.current_conv] = st.session_state.messages
    st.rerun()
