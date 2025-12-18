from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import os

class LocalRAG:
    def __init__(self, model_name="phi3:mini"):
        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize LLM
        self.llm = Ollama(model=model_name, temperature=0.2)
        
        # Vector store
        self.vectorstore = None
        self.qa_chain = None
        
    def load_documents(self, file_paths):
        """Load documents from various formats"""
        documents = []
        
        for path in file_paths:
            if path.endswith('.pdf'):
                loader = PyPDFLoader(path)
            else:
                loader = TextLoader(path)
            documents.extend(loader.load())
        
        return documents
    
    def create_vectorstore(self, documents, chunk_size=1250, chunk_overlap=250):
        """Split documents and create vector store"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        splits = text_splitter.split_documents(documents)
        
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        print(f"Created vectorstore with {len(splits)} chunks")
        
    def setup_qa_chain(self, k=3):
        """Setup the QA chain with retrieval"""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Call create_vectorstore first.")
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    
    def query(self, question):
        """Query the RAG system"""
        if self.qa_chain is None:
            raise ValueError("QA chain not setup. Call setup_qa_chain first.")
        
        # Get retrieved documents
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})  # Reduce from 5 to 3
        retrieved_docs = retriever.get_relevant_documents(question)
        
        # Build context and check size
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Estimate tokens (rough: 1 token ≈ 4 characters)
        estimated_tokens = len(context) / 4
        print(f"Context: {len(context)} chars (~{int(estimated_tokens)} tokens)")
        
        # Limit to ~2000 tokens for context (leave room for prompt + answer)
        max_chars = 8000  # ~2000 tokens
        if len(context) > max_chars:
            print(f"Context too large! Truncating from {len(context)} to {max_chars} chars")
            context = context[:max_chars]
        
        # Generate answer with timeout
        result = self.qa_chain.invoke({"query": question})
        
        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }
    
    def load_existing_vectorstore(self):
        """Load previously created vectorstore"""
        pass
        # if os.path.exists("./chroma_db"):
        #     self.vectorstore = Chroma(
        #         persist_directory="./chroma_db",
        #         embedding_function=self.embeddings
        #     )
        #     print("Loaded existing vectorstore")
        #     return True
        # return False

# Usage example
if __name__ == "__main__":
    # Initialize
    rag = LocalRAG()
    
    # Load your documents
    docs = rag.load_documents(["OmkarLetter.pdf"])
    
    # Create vectorstore
    rag.create_vectorstore(docs)
    
    # Setup QA chain
    rag.setup_qa_chain(k=3)
    
    # Query
    response = rag.query("What is the main topic of the document?")
    print(f"Answer: {response['answer']}")