import streamlit as st
from agent import query_agent
from retriever import store_docs, list_all_indexes, reset_vectorstore

# --- Page Setup ---
st.set_page_config(page_title="SmartLegalGPT", layout="wide")
st.title("⚖️ SmartLegalGPT")
st.caption("Ask legal questions from your uploaded PDFs using RAG + LLM")

# --- Sidebar: Upload & Select Document ---
st.sidebar.header("📄 Upload & Manage PDFs")
uploaded_file = st.sidebar.file_uploader("Upload a legal PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing..."):
        result = store_docs(uploaded_file)
        st.sidebar.success(f"Stored {result['documents']} chunks from `{result['index_name']}.pdf`")

available_indexes = list_all_indexes()
selected_index = st.sidebar.selectbox("📚 Choose Document", available_indexes) if available_indexes else None

# --- Sidebar: Reset Option ---
if st.sidebar.button("❌ Reset All Documents"):
    reset_vectorstore()
    st.sidebar.warning("All stored documents have been deleted.")

# --- Chat State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Chat UI ---
if selected_index:
    query = st.chat_input("Ask a legal question based on the selected document...")

    if query:
        with st.spinner("Thinking..."):
            answer = query_agent(query, selected_index)
            st.session_state.messages.append(("user", query))
            st.session_state.messages.append(("ai", answer))

    # Display chat history
    for sender, message in st.session_state.messages:
        st.chat_message(sender).write(message)
else:
    st.info("📂 Please upload and select a legal document to begin.")
