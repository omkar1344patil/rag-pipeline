from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.llms.base import LLM
from typing import Optional, List, Any
import requests
import json
import os
import time
from datetime import datetime

global model_name 

model_name = 'google/gemma-3-27b-it:free'

class OpenRouterLLM(LLM):
    """Custom LLM wrapper for OpenRouter API"""
    
    # model: str = "tngtech/deepseek-r1t2-chimera:free"
    # model: str = "google/gemma-3-12b-it:free"
    model: str = model_name
    api_key: str = ""
    temperature: float = 0.3
    max_tokens: int = 2000
    
    @property
    def _llm_type(self) -> str:
        return "openrouter"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Call OpenRouter API"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "RAG Pipeline",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["choices"][0]["message"]["content"]

class LocalRAG:
    def __init__(self, model_name=model_name, debug=False):
        self.debug = debug
        self.log("=" * 80)
        self.log("INITIALIZING RAG PIPELINE")
        self.log("=" * 80)
        
        # Initialize embedding model
        self.log("\n[1/3] Loading Embedding Model...")
        start = time.time()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.log(f"✓ Embedding model loaded in {time.time() - start:.2f}s")
        self.log(f"  Model: sentence-transformers/all-MiniLM-L6-v2")
        self.log(f"  Dimension: 384")
        
        # Initialize LLM (OpenRouter with DeepSeek)
        self.log("\n[2/3] Connecting to OpenRouter LLM...")
        start = time.time()
        
        # Get API key from environment
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment variables.\n"
                "Set it with: export OPENROUTER_API_KEY='your-key-here'"
            )
        
        self.llm = OpenRouterLLM(
            model=model_name,
            api_key=openrouter_api_key,
            temperature=0.3,
            max_tokens=2000
        )
        
        self.log(f"✓ LLM connected in {time.time() - start:.2f}s")
        self.log(f"  Model: {model_name}")
        self.log(f"  Temperature: 0.3")
        self.log(f"  Provider: OpenRouter")
        
        # Vector store
        self.vectorstore = None
        self.qa_chain = None
        
        self.log("\n[3/3] RAG Pipeline Ready!")
        self.log("=" * 80 + "\n")
        
    def log(self, message):
        """Print debug messages with timestamp"""
        if self.debug:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] {message}")
    
    def load_documents(self, file_paths):
        """Load documents from various formats"""
        self.log("\n" + "=" * 80)
        self.log("LOADING DOCUMENTS")
        self.log("=" * 80)
        
        documents = []
        
        for idx, path in enumerate(file_paths, 1):
            self.log(f"\n[{idx}/{len(file_paths)}] Processing: {os.path.basename(path)}")
            start = time.time()
            
            if path.endswith('.pdf'):
                loader = PyPDFLoader(path)
            else:
                loader = TextLoader(path)
            
            docs = loader.load()
            documents.extend(docs)
            
            self.log(f"✓ Loaded in {time.time() - start:.2f}s")
            self.log(f"  Pages/Sections: {len(docs)}")
            total_chars = sum(len(doc.page_content) for doc in docs)
            self.log(f"  Total characters: {total_chars:,}")
            
            if self.debug and docs:
                self.log(f"\n  Preview of first document:")
                preview = docs[0].page_content[:200].replace('\n', ' ')
                self.log(f"  '{preview}...'")
        
        self.log(f"\n✓ Total documents loaded: {len(documents)}")
        self.log("=" * 80 + "\n")
        return documents
    
    def create_vectorstore(self, documents, chunk_size=1000, chunk_overlap=200):
        """Split documents and create vector store"""
        self.log("\n" + "=" * 80)
        self.log("CREATING VECTOR STORE")
        self.log("=" * 80)
        
        # Split documents
        self.log(f"\n[1/3] Splitting documents...")
        self.log(f"  Chunk size: {chunk_size}")
        self.log(f"  Chunk overlap: {chunk_overlap}")
        
        start = time.time()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        self.log(f"✓ Split into {len(splits)} chunks in {time.time() - start:.2f}s")
        
        if self.debug and splits:
            avg_chunk_size = sum(len(s.page_content) for s in splits) / len(splits)
            self.log(f"  Average chunk size: {avg_chunk_size:.0f} characters")
            self.log(f"\n  Sample chunk:")
            self.log(f"  '{splits[0].page_content[:150]}...'")
        
        # Create embeddings
        self.log(f"\n[2/3] Creating embeddings...")
        start = time.time()
        
        # Show embedding example for first chunk
        if self.debug and splits:
            sample_text = splits[0].page_content[:100]
            sample_embedding = self.embeddings.embed_query(sample_text)
            self.log(f"  Sample text: '{sample_text}...'")
            self.log(f"  Embedding vector (first 5 dims): {sample_embedding[:5]}")
            self.log(f"  Embedding dimension: {len(sample_embedding)}")
        
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        self.log(f"✓ Embeddings created in {time.time() - start:.2f}s")
        
        # Persist
        self.log(f"\n[3/3] Persisting to disk...")
        self.vectorstore.persist()
        
        db_size = self._get_db_size()
        self.log(f"✓ Vector database saved to ./chroma_db")
        self.log(f"  Database size: {db_size}")
        self.log(f"  Total chunks stored: {len(splits)}")
        
        self.log("=" * 80 + "\n")
        
    def _get_db_size(self):
        """Calculate size of chroma_db directory"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk("./chroma_db"):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        
        if total_size < 1024:
            return f"{total_size} bytes"
        elif total_size < 1024 * 1024:
            return f"{total_size / 1024:.2f} KB"
        else:
            return f"{total_size / (1024 * 1024):.2f} MB"
    
    def setup_qa_chain(self, k=5):
        """Setup the QA chain with retrieval"""
        self.log("\n" + "=" * 80)
        self.log("SETTING UP QA CHAIN")
        self.log("=" * 80)
        
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Call create_vectorstore first.")
        
        self.log(f"\nConfiguring retriever...")
        self.log(f"  k (documents to retrieve): {k}")
        
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        self.log(f"\n✓ QA Chain configured")
        self.log("=" * 80 + "\n")
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    
    def query(self, question):
        """Query the RAG system with detailed debugging"""
        self.log("\n" + "=" * 80)
        self.log("QUERY PROCESSING")
        self.log("=" * 80)
        
        if self.qa_chain is None:
            raise ValueError("QA chain not setup. Call setup_qa_chain first.")
        
        self.log(f"\nUser Question: '{question}'")
        
        # Step 1: Embed the question
        self.log(f"\n[1/4] Embedding question...")
        start = time.time()
        question_embedding = self.embeddings.embed_query(question)
        embed_time = time.time() - start
        
        self.log(f"✓ Question embedded in {embed_time:.3f}s")
        if self.debug:
            self.log(f"  Embedding vector (first 5 dims): {question_embedding[:5]}")
        
        # Step 2: Retrieve relevant documents
        self.log(f"\n[2/4] Retrieving relevant chunks from vector DB...")
        start = time.time()
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.get_relevant_documents(question)
        retrieval_time = time.time() - start
        
        self.log(f"✓ Retrieved {len(retrieved_docs)} chunks in {retrieval_time:.3f}s")
        
        if self.debug:
            self.log(f"\n  Retrieved chunks:")
            for idx, doc in enumerate(retrieved_docs, 1):
                preview = doc.page_content[:100].replace('\n', ' ')
                self.log(f"  [{idx}] '{preview}...'")
                self.log(f"      Source: {doc.metadata.get('source', 'Unknown')}")
                self.log(f"      Length: {len(doc.page_content)} chars")
        
        # Step 3: Build context and prompt
        self.log(f"\n[3/4] Building context for LLM...")
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        context_length = len(context)
        self.log(f"✓ Context assembled: {context_length} characters")
        
        if self.debug:
            self.log(f"\n  Full context preview:")
            self.log(f"  '{context[:200]}...'")
        
        # Step 4: Generate answer
        self.log(f"\n[4/4] Generating answer with LLM...")
        start = time.time()
        result = self.qa_chain.invoke({"query": question})
        generation_time = time.time() - start
        
        self.log(f"✓ Answer generated in {generation_time:.3f}s")
        self.log(f"\n  Answer: '{result['result'][:150]}...'")
        
        # Summary
        total_time = embed_time + retrieval_time + generation_time
        self.log(f"\n" + "-" * 80)
        self.log(f"PERFORMANCE SUMMARY:")
        self.log(f"  Embedding:    {embed_time:.3f}s ({embed_time/total_time*100:.1f}%)")
        self.log(f"  Retrieval:    {retrieval_time:.3f}s ({retrieval_time/total_time*100:.1f}%)")
        self.log(f"  Generation:   {generation_time:.3f}s ({generation_time/total_time*100:.1f}%)")
        self.log(f"  TOTAL:        {total_time:.3f}s")
        self.log("=" * 80 + "\n")
        
        return {
            "answer": result["result"],
            "sources": result["source_documents"],
            "metrics": {
                "embed_time": embed_time,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "chunks_retrieved": len(retrieved_docs),
                "context_length": context_length
            }
        }
    
    def load_existing_vectorstore(self):
        """Load previously created vectorstore"""
        self.log("\n" + "=" * 80)
        self.log("LOADING EXISTING VECTOR STORE")
        self.log("=" * 80)
        
        if os.path.exists("./chroma_db"):
            start = time.time()
            self.vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=self.embeddings
            )
            
            db_size = self._get_db_size()
            self.log(f"\n✓ Loaded existing vectorstore in {time.time() - start:.2f}s")
            self.log(f"  Location: ./chroma_db")
            self.log(f"  Size: {db_size}")
            self.log("=" * 80 + "\n")
            return True
        else:
            self.log("\n✗ No existing vectorstore found")
            self.log("=" * 80 + "\n")
            return False

# Example usage with debug mode
if __name__ == "__main__":
    # Initialize with debug=True
    rag = LocalRAG(debug=True)
    
    # Load documents
    docs = rag.load_documents(["bigdatatutorial.pdf"])
    
    # Create vectorstore
    rag.create_vectorstore(docs, chunk_size=1000, chunk_overlap=200)
    
    # Setup QA chain
    rag.setup_qa_chain(k=5)
    
    # Query
    response = rag.query("What is the main topic of the document?")
    
    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(response["answer"])
    print("\nMetrics:", response["metrics"])