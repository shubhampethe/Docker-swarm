import os
import sys
import tempfile
import asyncio
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

from rag.pipeline import RAGPipeline
from rag.config import settings

# Initialize pipeline in session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

pipeline: RAGPipeline = st.session_state.pipeline

# Custom CSS for polished, modern UI
st.markdown("""
<style>
    /* Metric Cards */
    .metric-box {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        border: 1px solid #313244;
    }
    .metric-title {
        color: #a6adc8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-val {
        color: #cdd6f4;
        font-size: 1.1rem;
        font-weight: 600;
    }
    /* Citation Box */
    .citation-box {
        background: #181825;
        border-left: 3px solid #89b4fa;
        padding: 8px 12px;
        margin: 6px 0;
        font-size: 0.85rem;
        border-radius: 0 6px 6px 0;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.title("⚡ RAG Control Panel")
    st.caption("LlamaParse (Async) • Embedding Container • Chroma • Groq")

    # 1. System Health Status
    st.subheader("System Status")
    health = pipeline.health_check()

    emb_ok = health["embedding_container"]["reachable"]
    llama_ok = health["llamaparse_parser"]["configured"]
    chroma_docs = health["chroma_local_store"]["total_documents"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Embedding API</div>
            <div class="metric-val" style="color: {'#a6e3a1' if emb_ok else '#f38ba8'}">
                {'● Online (8001)' if emb_ok else '● Offline'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Indexed Chunks</div>
            <div class="metric-val" style="color: #89b4fa">{chroma_docs}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">LlamaParse Service</div>
        <div class="metric-val" style="color: {'#a6e3a1' if llama_ok else '#f9e2af'}">
            {'● Async API Active' if llama_ok else '● Local Fallback'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    swarm_info = health["swarm_autoscaler"]
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">Swarm Auto-Scaler</div>
        <div class="metric-val" style="color: {'#a6e3a1' if swarm_info['active'] else '#6c7086'}">
            {'● Active (1 ➔ 3 Replicas)' if swarm_info['active'] else '● Inactive (1 Container)'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. Document Ingestion Section
    st.subheader("📄 Upload & Ingest")
    uploaded_file = st.file_uploader(
        "Choose a document (PDF, DOCX, TXT, MD)",
        type=["pdf", "docx", "txt", "md"],
        help="LlamaParse extracts high-accuracy text, tables, and layouts asynchronously."
    )

    if uploaded_file is not None:
        if st.button("🚀 Ingest Document (Async)", use_container_width=True, type="primary"):
            with st.spinner("Processing document with Async LlamaParse & Embedding Container..."):
                # Save to temporary file
                file_ext = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    # Run async ingestion
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(pipeline.aingest_file(tmp_path))

                    st.success(f"**Successfully Ingested!**\n- File: `{uploaded_file.name}`\n- Chunks Created: `{result['chunks_created']}`\n- Indexed into Chroma: `{result['chunks_indexed']}`")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

    st.markdown("---")

    # 3. Settings Drawer
    st.subheader("⚙️ Query Configuration")
    top_k = st.slider("Retrieved Context Chunks (Top-K)", min_value=1, max_value=8, value=settings.retrieval_top_k)
    
    groq_model = st.selectbox(
        "Groq LLM Model",
        options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        index=0
    )
    settings.groq_model = groq_model

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ==========================================
# MAIN INTERFACE
# ==========================================
st.title("⚡ Enterprise RAG Assistant")
st.caption("Ask questions grounded in your ingested documents. Powered by **Groq ChatGroq** and **LlamaParse**.")

# Show welcome hints when conversation is empty
if not st.session_state.messages:
    if chroma_docs == 0:
        st.info("👋 Welcome! Upload a PDF or document in the left sidebar to begin querying your knowledge base.")
    else:
        st.info(f"📚 {chroma_docs} chunks are ready in your local Chroma database! Ask any question below.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"📚 Sources ({len(message['sources'])} chunks cited)"):
                for idx, src in enumerate(message["sources"], start=1):
                    st.markdown(f"""
                    <div class="citation-box">
                        <b>Chunk #{idx}</b> • <i>{src['source']} (Page {src['page']})</i><br/>
                        <span style="color: #a6adc8;">{src['preview']}</span>
                    </div>
                    """, unsafe_allow_html=True)

# Chat Input & Response Generation
if prompt := st.chat_input("Ask a question about your ingested documents..."):
    # Render user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving semantic context & generating answer with Groq..."):
            try:
                result = pipeline.query(prompt, top_k=top_k)
                answer_text = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer_text)

                if sources:
                    with st.expander(f"📚 Sources ({len(sources)} chunks cited)"):
                        for idx, src in enumerate(sources, start=1):
                            st.markdown(f"""
                            <div class="citation-box">
                                <b>Chunk #{idx}</b> • <i>{src['source']} (Page {src['page']})</i><br/>
                                <span style="color: #a6adc8;">{src['preview']}</span>
                            </div>
                            """, unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources
                })
            except Exception as e:
                error_msg = f"⚠️ Query execution failed: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
