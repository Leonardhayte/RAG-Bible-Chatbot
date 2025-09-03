import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from google import genai
from dotenv import load_dotenv

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(page_title="RAG Chat", page_icon="📖", layout="wide")

# -------------------------------
# Session state
# -------------------------------
if "messages" not in st.session_state: st.session_state.messages = []
if "model" not in st.session_state: st.session_state.model = None
if "index" not in st.session_state: st.session_state.index = None
if "chunks" not in st.session_state: st.session_state.chunks = []

# -------------------------------
# Gemini API
# -------------------------------
load_dotenv("api.env")  # load key from api.env
client = genai.Client()

# -------------------------------
# Load SentenceTransformer model
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Build FAISS index with overlap
# -------------------------------
@st.cache_data(show_spinner=False)
def build_index(text, chunk_size=3000, overlap=400):
    """
    Split text into overlapping chunks to preserve verse flow.
    Larger chunk size + overlap improves context.
    """
    model = load_model()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    st.info(f"Building embeddings for {len(chunks)} chunks, please wait...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    # Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return model, index, chunks

# -------------------------------
# Retrieve relevant chunks
# -------------------------------
def retrieve(query, top_k=10):
    if st.session_state.index is None or len(st.session_state.chunks) == 0:
        return ["Biblia haijapakiwa!"], None
    model = load_model()
    query_emb = model.encode([query])
    query_emb = np.array(query_emb, dtype="float32")
    distances, indices = st.session_state.index.search(query_emb, top_k)
    results = [st.session_state.chunks[i] for i in indices[0]]
    return results, distances

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
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("📖 Bible Document")
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []

# -------------------------------
# Load Bible and build index
# -------------------------------
if st.session_state.index is None:
    try:
        with open("bible.txt", "r", encoding="utf-8") as f:
            text = f.read()
        st.session_state.model, st.session_state.index, st.session_state.chunks = build_index(text)
        st.success("Bible loaded, chunked, and indexed! You can start chatting.")
    except FileNotFoundError:
        st.error("bible.txt not found! Place it in the same folder as app.py.")

# -------------------------------
# Page title
# -------------------------------
st.title("📖 READ THE BIBLE (RAG Chatbot)")

# -------------------------------
# Chat window
# -------------------------------
chat_box = st.container()
with chat_box:
    if not st.session_state.messages:
        chat_bubble("Habari! Mimi niko tayari kukusaidia. Uliza chochote kuhusu Biblia.", "assistant")
    for msg in st.session_state.messages:
        chat_bubble(msg["content"], role=msg["role"])

# -------------------------------
# Input form
# -------------------------------
disabled = st.session_state.index is None
with st.form("chat_form", clear_on_submit=True):
    user_query = st.text_input("Andika swali lako hapa...", disabled=disabled)
    sent = st.form_submit_button("Tuma", disabled=disabled)

if sent and user_query:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Retrieve multiple chunks (more top_k for better coverage)
    retrieved_texts, _ = retrieve(user_query, top_k=10)
    context = "\n\n".join(retrieved_texts)

    # Build improved prompt
    prompt = f"""
You are a helpful Bible assistant.
Answer the question using the context below.
- Always include the **book name, chapter, and verse** when quoting.
- Write the answer in clear, natural Swahili.
- If the context partially answers the question, combine the information naturally.
- If there is no relevant information, reply: "Sina uhakika kwa maandiko haya."

Context:
{context}

Question: {user_query}

Answer:
"""

    # Generate answer
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"Samahani, hitilafu imetokea: {e}"

    # Save and render assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    chat_bubble(user_query, role="user")
    chat_bubble(answer, role="assistant")
