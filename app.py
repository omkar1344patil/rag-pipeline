import streamlit as st
import os
import shutil
# from rag_pipeline_copy import LocalRAG
from upgraded_rag_pipeline import LocalRAG

# Page config
st.set_page_config(
    page_title="Local RAG",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = LocalRAG(debug=True)
    st.session_state.documents_loaded = False
    st.session_state.chat_history = []

# Sidebar for data sources
with st.sidebar:
    st.title("🤖 Local RAG")
    
    tab1, tab2, tab3 = st.tabs(["Data Sources", "Settings", "About"])
    
    with tab1:
        st.header("Directly import your data")
        st.caption("Convert your data into embeddings for utilization during chat")
        
        # File upload section
        st.subheader("📁 Local Files")
        uploaded_files = st.file_uploader(
            "Select Files",
            type=['pdf', 'txt', 'csv', 'docx', 'md'],
            accept_multiple_files=True,
            help="Limit 200MB per file • CSV, DOCX, EPUB, IPYNB, JSON, MD, PDF, PPT, PPTX, TXT"
        )
        
        if uploaded_files:
            if st.button("Process Files", type="primary"):
                with st.spinner("Processing documents..."):
                    # Save uploaded files temporarily
                    temp_dir = "./temp_uploads"
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(file_path)
                    
                    # Process documents
                    try:
                        docs = st.session_state.rag.load_documents(file_paths)
                        st.session_state.rag.create_vectorstore(docs)
                        st.session_state.rag.setup_qa_chain(k=3)
                        st.session_state.documents_loaded = True
                        st.success(f"✅ Processed {len(uploaded_files)} files!")
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")
                    finally:
                        # Cleanup temp files
                        shutil.rmtree(temp_dir, ignore_errors=True)
        
        st.divider()
        
        # Clear database button
        if st.button("🗑️ Clear Vector Database", type="secondary"):
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
                st.session_state.rag = LocalRAG()
                st.session_state.documents_loaded = False
                st.session_state.chat_history = []
                st.success("Database cleared!")
                st.rerun()
        
        # Show current status
        if os.path.exists("./chroma_db"):
            st.info("📊 Vector database exists")
        else:
            st.warning("📊 No vector database found")
    
    with tab2:
        st.header("⚙️ Settings")
        
        # Model settings
        model = st.selectbox(
            "Model",
            ["phi3:mini", "llama2", "mistral"],
            help="Select the Ollama model to use"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Higher values make output more random"
        )
        
        k_docs = st.slider(
            "Number of retrieved documents",
            min_value=1,
            max_value=10,
            value=3,
            help="How many document chunks to retrieve"
        )
        
        if st.button("Apply Settings"):
            st.session_state.rag = LocalRAG(model_name=model)
            st.session_state.rag.llm.temperature = temperature
            if st.session_state.documents_loaded:
                st.session_state.rag.load_existing_vectorstore()
                st.session_state.rag.setup_qa_chain(k=k_docs)
            st.success("Settings applied!")
    
    with tab3:
        st.header("ℹ️ About")
        st.markdown("""
        **Local RAG Pipeline**
        
        Built with:
        - 🦜 LangChain
        - 🤖 Ollama (Phi-3-mini)
        - 📊 ChromaDB
        - 🎨 Streamlit
        
        Features:
        - Upload multiple documents
        - Persistent vector storage
        - Multi-turn conversations
        - Source attribution
        """)

# Main chat interface
st.title("💬 Chat with your documents")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.caption(f"**Source {i}:** {source.metadata.get('source', 'Unknown')}")
                    st.text(source.page_content[:300] + "...")

# Chat input
if prompt := st.chat_input("How can I help?", disabled=not st.session_state.documents_loaded):
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag.query(prompt)
                st.markdown(response["answer"])
                
                # Show sources
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(response["sources"], 1):
                        st.caption(f"**Source {i}:** {source.metadata.get('source', 'Unknown')}")
                        st.text(source.page_content[:300] + "...")
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response["sources"]
                })
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Show message if no documents loaded
if not st.session_state.documents_loaded:
    st.info("👈 Upload documents from the sidebar to get started!")