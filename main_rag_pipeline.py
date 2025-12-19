from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Optional, List, Any
import requests
import json
import os
import time
from datetime import datetime


class OpenRouterLLM(LLM):
    """Custom LLM wrapper for OpenRouter API"""
    
    model: str = "google/gemma-3-27b-it:free"
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

class BaseRAG:
    """Base class with shared RAG functionality"""
    
    def __init__(self, debug=False):
        self.debug = debug
        self.vectorstore = None
        self.qa_chain = None
        
        self.log("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.log("✓ Embedding model loaded (384 dimensions)")
    
    def log(self, message):
        """Print debug messages"""
        if self.debug:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def load_documents(self, file_paths):
        """Load documents from various formats"""
        self.log(f"Loading {len(file_paths)} document(s)...")
        documents = []
        
        for path in file_paths:
            if path.endswith('.pdf'):
                loader = PyPDFLoader(path)
            else:
                loader = TextLoader(path)
            
            docs = loader.load()
            documents.extend(docs)
        
        self.log(f"✓ Loaded {len(documents)} document(s)")
        return documents
    
    def create_vectorstore(self, documents, chunk_size=1000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        splits = text_splitter.split_documents(documents)
        self.log(f"  Split into {len(splits)} chunks")

        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings
        )

        self.log(f"✓ Vector store created (in-memory)")
    
    def setup_qa_chain(self, k=5):
        """Setup QA chain with retrieval"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")

        self.log(f"Setting up QA chain (k={k})...")

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

        system_prompt = (
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Don't assume anything."
            "Be concise while using proper sentences and accurate. Show logic and brief explanation behind your reasoning if it's a complex question only.\n\n"
            "Context: {context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.qa_chain = create_retrieval_chain(self.retriever, question_answer_chain)

        self.log("✓ QA chain setup complete")
    
    def query(self, question):
        """Query the RAG system"""
        if self.qa_chain is None:
            raise ValueError("QA chain not setup")

        self.log(f"Querying: '{question[:50]}...'")

        result = self.qa_chain.invoke({"input": question})

        return {
            "answer": result["answer"],
            "sources": result.get("context", [])
        }
    
    def load_existing_vectorstore(self):
        self.log("No persistent storage on Streamlit Cloud")
        return False
    
    def clear_vectorstore(self):
        """Clear vector database properly"""
        import shutil

        self.log("Clearing vector database...")

        if self.vectorstore is not None:
            try:
                self.vectorstore = None
            except:
                pass
            
        if self.qa_chain is not None:
            self.qa_chain = None


        if os.path.exists("/tmp/chroma_db"):
            try:
                shutil.rmtree("/tmp/chroma_db")
                self.log("✓ Vector database cleared")
                return True
            except Exception as e:
                self.log(f"Error clearing database: {e}")
                return False
        else:
            self.log("No database to clear")
            return False

class OpenRouterRAG(BaseRAG):
    """RAG with Personal API"""
    
    def __init__(self, model_name="google/gemma-3-27b-it:free", api_key=None, debug=False):
        super().__init__(debug)
        
        self.log("=" * 60)
        self.log("INITIALIZING OPENROUTER RAG")
        self.log("=" * 60)
        
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        
        if not api_key:
            raise ValueError("OpenRouter API key required")
        
        self.log(f"Connecting to OpenRouter ({model_name})...")
        self.llm = OpenRouterLLM(
            model=model_name,
            api_key=api_key,
            temperature=0.3,
            max_tokens=2000
        )

        
        self.log(f"✓ OpenRouter connected: {model_name}")
        self.log("=" * 60)

class LocalRAG(BaseRAG):
    """RAG with local Ollama models"""
    
    def __init__(self, model_name="phi3:mini", debug=False):
        super().__init__(debug)
        
        self.log("=" * 60)
        self.log("INITIALIZING LOCAL RAG")
        self.log("=" * 60)
        
        self.log(f"Connecting to Ollama ({model_name})...")
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.3
        )
        
        self.log(f"✓ Ollama connected: {model_name}")
        self.log("=" * 60)


if __name__ == "__main__":
    print("\n=== Testing RAG ===")
    rag = OpenRouterRAG(
        model_name="google/gemma-3-27b-it:free",
        debug=True
    )
    
    docs = rag.load_documents(["test.pdf"])
    rag.create_vectorstore(docs)
    rag.setup_qa_chain(k=5)
    
    response = rag.query("What is this about?")
    print(f"\nAnswer: {response['answer']}")
    
    print("\n\n=== Testing Local RAG ===")
    local_rag = LocalRAG(model_name="phi3:mini", debug=True)
    
    docs = local_rag.load_documents(["test.pdf"])
    local_rag.create_vectorstore(docs)
    local_rag.setup_qa_chain(k=5)
    
    response = local_rag.query("What is this about?")
    print(f"\nAnswer: {response['answer']}")