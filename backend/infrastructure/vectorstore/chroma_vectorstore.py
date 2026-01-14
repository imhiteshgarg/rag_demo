"""ChromaDB vector store implementation."""
import sys
import os
from pathlib import Path
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from domain.entities.document import Document
from domain.repositories.vectorstore_repository import VectorStoreRepository


class ChromaVectorStoreRepository(VectorStoreRepository):
    """ChromaDB-based vector store repository implementation."""
    
    def __init__(self, embeddings, persist_directory: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the vector store repository.
        
        Args:
            embeddings: Embeddings object
            persist_directory: Directory to persist the vector database
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.vectorstore = None
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def create_or_load(self, documents: List[Document], force_recreate: bool = False) -> None:
        """Create or load the vector store."""
        if not force_recreate and os.path.exists(self.persist_directory):
            print("Loading existing vector database...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print("Vector database loaded successfully!")
        else:
            print("Creating new vector database from documents...")
            
            if not documents:
                print("No documents provided.")
                return
            
            print(f"Loaded {len(documents)} documents")
            
            # Convert domain documents to LangChain documents
            try:
                from langchain_core.documents import Document as LangChainDocument
            except ImportError:
                try:
                    from langchain.schema import Document as LangChainDocument
                except ImportError:
                    from langchain.docstore.document import Document as LangChainDocument
            langchain_docs = []
            for doc in documents:
                langchain_docs.append(LangChainDocument(
                    page_content=doc.content,
                    metadata=doc.metadata or {}
                ))
            
            # Split documents into chunks
            texts = self.text_splitter.split_documents(langchain_docs)
            print(f"Split into {len(texts)} text chunks")
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print("Vector database created successfully!")
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents."""
        if not self.vectorstore:
            return []
        
        try:
            # Use similarity search
            results = self.vectorstore.similarity_search(query, k=k)
            
            # Convert LangChain documents to domain documents
            documents = []
            for result in results:
                documents.append(Document(
                    content=result.page_content,
                    metadata=result.metadata if hasattr(result, 'metadata') else {}
                ))
            
            return documents
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []
    
    def is_initialized(self) -> bool:
        """Check if vector store is initialized."""
        return self.vectorstore is not None
