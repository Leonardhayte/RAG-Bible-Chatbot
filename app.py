import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from google import genai
from dotenv import load_dotenv
import pickle

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(page_title="RAG Bible Chat", page_icon="📖", layout="wide")

# -------------------------------
# Session state
# -------------------------------
if "messages" not in st.session_state: 
    st.session_state.messages = []
if "model" not in st.session_state: 
    st.session_state.model = None
if "indexes" not in st.session_state: 
    st.session_state.indexes = {}

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
# Build or load TWO indexes (small + large)
# -------------------------------
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
            st.info(f"🔄 Loading {name} index...")
            index = faiss.read_index(index_file)
            with open(chunk_file, "rb") as f:
                chunks = pickle.load(f)
        else:
            st.info(f"⚙️ Building {name} index...")
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
    # Detect if query looks like a verse (e.g. "Genesis 1:1")
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
# Build final prompt (with memory)
# -------------------------------
def build_prompt(user_query, context):
    history = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]]
    )
    return f"""
You are a Bible assistant. Answer in clear Swahili.
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
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("📖 Bible Document")
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []

# -------------------------------
# Load Bible and build indexes
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
# Page title
# -------------------------------
st.title("📖 READ THE BIBLE (RAG Chatbot)")

# -------------------------------
# Chat window
# -------------------------------
chat_box = st.container()
with chat_box:
    if not st.session_state.messages:
        chat_bubble("Habari! Niko tayari kukusaidia kuhusu Biblia.", "assistant")
    for msg in st.session_state.messages:
        chat_bubble(msg["content"], role=msg["role"])

# -------------------------------
# Input form BELOW chat
# -------------------------------
disabled = not st.session_state.indexes
with st.form("chat_form", clear_on_submit=True):
    user_query = st.text_input("✍️ Andika swali lako hapa...", disabled=disabled)
    sent = st.form_submit_button("Tuma", disabled=disabled)

if sent and user_query:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Retrieve relevant context
    retrieved_texts = retrieve(user_query, top_k=5)
    context = "\n\n".join(retrieved_texts)

    # Build final prompt
    qa_prompt = build_prompt(user_query, context)

    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=qa_prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"Samahani, hitilafu imetokea: {e}"

    # Save and render assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    chat_bubble(user_query, role="user")
    chat_bubble(answer, role="assistant")
