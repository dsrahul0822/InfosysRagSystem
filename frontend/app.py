import os
import sys
from pathlib import Path
import streamlit as st

# ------------------------------------------------------------
# Make sure imports + relative paths work (NO backend change)
# ------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]   # project root (same level as main.py)
sys.path.insert(0, str(ROOT_DIR))                # allow: from main import RAGService
os.chdir(ROOT_DIR)                               # ensure PDF_PATH / chroma_db relative paths work

from main import RAGService  # uses your existing backend as-is


# ------------------------------------------------------------
# Streamlit Page Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Employee Handbook HR Bot",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Employee Handbook HR Bot")
st.caption("Ask questions from your Employee Handbook PDF using your existing RAG backend.")


# ------------------------------------------------------------
# Load RAG service once (cached)
# ------------------------------------------------------------
@st.cache_resource
def get_rag_service():
    return RAGService()


# ------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    # Helpful status checks (no changes to backend)
    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    st.write("**OPENAI_API_KEY:**", "✅ Found" if api_key_present else "❌ Missing")

    st.divider()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown(
        """
**Notes**
- In Streamlit Cloud, set `OPENAI_API_KEY` in **App → Settings → Secrets**.
- Your backend will create/load `chroma_db` automatically.
        """.strip()
    )


# ------------------------------------------------------------
# Session State: Chat History
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything from the Employee Handbook."}
    ]


# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ------------------------------------------------------------
# Chat Input
# ------------------------------------------------------------
user_query = st.chat_input("Type your question (e.g., 'How many paid leaves do I get?')")

if user_query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                rag = get_rag_service()
                answer = rag.ask(user_query)  # backend call (as-is)
            except Exception as e:
                answer = f"⚠️ Error: {e}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
