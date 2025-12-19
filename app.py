"""
Streamlit UI for RAG System
"""
import streamlit as st
import os
import shutil
from upgraded_rag_pipeline import OpenRouterRAG, LocalRAG
import time


st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="🔍",
    layout="wide"
)


if 'rag' not in st.session_state:
    st.session_state.rag = None
    st.session_state.documents_loaded = False
    st.session_state.chat_history = []
    st.session_state.llm_type = "openrouter"
    st.session_state.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
    st.session_state.openrouter_model = "google/gemma-3-27b-it:free"
    st.session_state.local_model = "phi3:mini"
    st.session_state.k_docs = 3


with st.sidebar:
    st.markdown("# 🔍 RAG Pipeline")
    
    tab1, tab2, tab3 = st.tabs(["📁 Data", "⚙️ Settings", "ℹ️ Info"])
    
    with tab1:
        st.header("Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'txt', 'md', 'csv'],
            accept_multiple_files=True,
            help="Upload documents to index"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                with st.spinner("Processing documents..."):
                    os.makedirs("./uploads", exist_ok=True)
                    file_paths = []
                    
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join("./uploads", uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(file_path)
                    
                    try:
                        if st.session_state.rag is None:
                            st.error("⚠️ Please configure LLM settings first!")
                        else:
                            docs = st.session_state.rag.load_documents(file_paths)
                            st.session_state.rag.create_vectorstore(docs)
                            st.session_state.rag.setup_qa_chain(k=st.session_state.k_docs)
                            st.session_state.documents_loaded = True
                            st.success(f"✅ Processed {len(uploaded_files)} files!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                    finally:
                        shutil.rmtree("./uploads", ignore_errors=True)
        
        st.divider()
        
        if os.path.exists("./chroma_db"):
            st.info("📊 Vector database active")
            if st.button("🗑️ Clear Database", use_container_width=True):
                if st.session_state.rag:
                    success = st.session_state.rag.clear_vectorstore()
                    if success:
                        st.session_state.documents_loaded = False
                        st.session_state.chat_history = []
                        st.success("✅ Database cleared!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to clear database")
                else:
                    # Fallback if no RAG instance
                    import shutil
                    if os.path.exists("./chroma_db"):
                        shutil.rmtree("./chroma_db")
                        st.session_state.documents_loaded = False
                        st.session_state.chat_history = []
                        st.rerun()
                    else:
                        st.warning("📊 No vector database")
    
    with tab2:
        st.header("⚙️ LLM Settings")
        
        # LLM Type Selection
        llm_type = st.radio(
            "Select LLM Provider",
            ["Personal API", "Local LLM"],
            index=0 if st.session_state.llm_type == "openrouter" else 1
        )
        
        st.divider()
        
        # OpenRouter Settings
        if llm_type == "Personal API":
            st.subheader("🌐 Personal API Key Configuration")
            
            openrouter_api_key = st.text_input(
                "Enter your API Key",
                value=st.session_state.openrouter_api_key,
                type="password",
                help="Input your API key here"
            )
            
            openrouter_model = st.text_input(
                "Model Name (optional)",
                value=st.session_state.openrouter_model,
                help="Example: google/gemma-3-27b-it:free"
            )
            
            
        # Local LLM Settings
        else:
            st.subheader("🖥️ Local LLM Configuration")
            
            local_models = [
                "phi3:mini",
                "gemma2:7b", 
                "llama3.2:3b",
                "mistral:7b",
                "qwen2.5:7b"
            ]
            
            model_selection = st.selectbox(
                "Select Model",
                options=local_models + ["Custom..."],
                index=local_models.index(st.session_state.local_model) if st.session_state.local_model in local_models else 0
            )
            
            if model_selection == "Custom...":
                local_model = st.text_input(
                    "Custom Model Name",
                    value=st.session_state.local_model if st.session_state.local_model not in local_models else "",
                    placeholder="model:tag"
                )
            else:
                local_model = model_selection
            
            st.caption("💡 Run: `ollama pull model-name`")
        
        st.divider()
        
        # Retrieval Settings
        st.subheader("📊 Retrieval Settings")
        
        k_docs = st.slider(
            "Documents to Retrieve",
            min_value=1,
            max_value=10,
            value=st.session_state.k_docs
        )
        
        st.divider()
        
        # Apply Button
        if st.button("✅ Apply Settings", type="primary", use_container_width=True):
            try:
                st.session_state.k_docs = k_docs
                
                if llm_type == "Personal API":
                    if not openrouter_api_key:
                        st.error("❌ API key required")
                    else:
                        st.session_state.llm_type = "openrouter"
                        st.session_state.openrouter_api_key = openrouter_api_key
                        st.session_state.openrouter_model = openrouter_model
                        
                        with st.spinner("Initializing your API Key..."):
                            st.session_state.rag = OpenRouterRAG(
                                model_name=openrouter_model,
                                api_key=openrouter_api_key,
                                debug=False
                            )
                        st.success(f"✅ Your API key initiated: {openrouter_model}")
                else:
                    st.session_state.llm_type = "local"
                    st.session_state.local_model = local_model
                    
                    with st.spinner(f"Initializing {local_model}..."):
                        st.session_state.rag = LocalRAG(
                            model_name=local_model,
                            debug=False
                        )
                    st.success(f"✅ Local: {local_model}")
                
                # Load existing vectorstore
                if os.path.exists("./chroma_db"):
                    st.session_state.rag.load_existing_vectorstore()
                    st.session_state.rag.setup_qa_chain(k=k_docs)
                    st.session_state.documents_loaded = True
                
            except Exception as e:
                st.error(f"❌ {str(e)}")
    
    with tab3:
        st.header("ℹ️ About")
        st.markdown("""
        **RAG Pipeline**
        
        Retrieval-Augmented Generation with:
        - OpenRouter API
        - Local Ollama models
        - ChromaDB vector store
        - Multi-turn chat
        """)
        
        if st.session_state.rag:
            st.divider()
            st.caption("**Current Config:**")
            st.caption(f"Type: {st.session_state.llm_type}")
            if st.session_state.llm_type == "openrouter":
                st.caption(f"Model: {st.session_state.openrouter_model}")
            else:
                st.caption(f"Model: {st.session_state.local_model}")
            st.caption(f"Retrieval: k={st.session_state.k_docs}")

# Main
st.markdown("# RAG Pipeline")
st.markdown("### Import your data and chat with your documents")

if st.session_state.rag is None:
    st.info("👈 Configure LLM settings in sidebar")
    st.stop()

# Chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.caption(f"**[{i}]** {src.metadata.get('source', 'Unknown')}")
                    st.text(src.page_content[:200] + "...")

# Chat input
if prompt := st.chat_input("Ask about your documents...", disabled=not st.session_state.documents_loaded):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag.query(prompt)
                st.markdown(response["answer"])
                
                with st.expander("📚 Sources"):
                    for i, src in enumerate(response["sources"], 1):
                        st.caption(f"**[{i}]** {src.metadata.get('source', 'Unknown')}")
                        st.text(src.page_content[:200] + "...")
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response["sources"]
                })
            except Exception as e:
                st.error(f"Error: {str(e)}")

if not st.session_state.documents_loaded:
    st.info("👈 Upload documents to start")