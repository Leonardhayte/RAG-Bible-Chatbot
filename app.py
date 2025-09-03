import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from google import genai

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
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
client = genai.Client()

# -------------------------------
# Load SentenceTransformer model
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Build FAISS index with caching
# -------------------------------
@st.cache_data(show_spinner=False)
def build_index(text, chunk_size=4000):
    """
    Split text into bigger chunks (default 4000 chars) to reduce embedding time.
    Return model, FAISS index, and chunks.
    """
    model = load_model()
    
    # Split text into chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Generate embeddings
    st.info(f"Building embeddings for {len(chunks)} chunks, please wait...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    
    # Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return model, index, chunks

# -------------------------------
# Retrieve relevant chunk
# -------------------------------
def retrieve(query, top_k=1):
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
            text-align: left;">
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
# Load Bible and build index if not already
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
    # Show initial greeting immediately
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
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Retrieve relevant chunk(s)
    retrieved_text, _ = retrieve(user_query)
    
    # Build prompt for Gemini
    prompt = f"""
You are an assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say you don't know.

Context:
{retrieved_text[0]}

Question: {user_query}
"""
    # Generate answer
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"Samahani, hitilafu imetokea: {e}"
    
    # Store assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Render chat
    chat_bubble(user_query, role="user")
    chat_bubble(answer, role="assistant")
