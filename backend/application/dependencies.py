"""Dependency injection container."""
import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.embeddings.sentence_transformers import SentenceTransformersEmbeddings
from infrastructure.vectorstore.chroma_vectorstore import ChromaVectorStoreRepository
from infrastructure.document_loaders.langchain_loader import LangChainDocumentRepository
from infrastructure.llm.simple_llm import SimpleLLMRepository
from use_cases.query_rag import QueryRAGUseCase
from use_cases.initialize_rag import InitializeRAGUseCase
from interfaces.controllers.rag_controller import RAGController


class DIContainer:
    """Dependency Injection Container."""
    
    def __init__(self, documents_dir: str, persist_directory: str):
        """
        Initialize the container with dependencies.
        
        Args:
            documents_dir: Directory containing documents
            persist_directory: Directory for vector store persistence
        """
        self.documents_dir = documents_dir
        self.persist_directory = persist_directory
        
        # Initialize infrastructure (lazy-load embeddings to save memory)
        self._embeddings = None  # Lazy-loaded
        self._document_repo = LangChainDocumentRepository()
        self._vectorstore_repo = None  # Will be created when needed
        self._persist_directory = persist_directory
        self._llm_repo = SimpleLLMRepository()
        self._documents_dir = documents_dir
        
        # Use cases will be lazy-loaded when vectorstore is created
        self._init_use_case = None
        self._query_use_case = None
        self._rag_controller = None
    
    def _ensure_use_cases(self):
        """Ensure use cases are initialized (lazy-loaded)."""
        if self._init_use_case is None:
            self._init_use_case = InitializeRAGUseCase(
                document_repo=self._document_repo,
                vectorstore_repo=self.vectorstore_repo,  # This will lazy-load vectorstore
                documents_dir=self._documents_dir
            )
            self._query_use_case = QueryRAGUseCase(
                vectorstore_repo=self.vectorstore_repo,
                llm_repo=self._llm_repo
            )
            self._rag_controller = RAGController(
                query_use_case=self._query_use_case,
                init_use_case=self._init_use_case
            )
    
    @property
    def rag_controller(self) -> RAGController:
        """Get RAG controller (lazy-loaded)."""
        self._ensure_use_cases()
        return self._rag_controller
    
    @property
    def init_use_case(self) -> InitializeRAGUseCase:
        """Get initialization use case (lazy-loaded)."""
        self._ensure_use_cases()
        return self._init_use_case
    
    def _get_embeddings(self):
        """Lazy-load embeddings model only when needed."""
        if self._embeddings is None:
            print("Loading embeddings model (this may take a moment)...")
            self._embeddings = SentenceTransformersEmbeddings()
            print("Embeddings model loaded.")
        return self._embeddings
    
    @property
    def vectorstore_repo(self):
        """Get vectorstore repository (lazy-loaded)."""
        if self._vectorstore_repo is None:
            print("Creating vectorstore repository...")
            self._vectorstore_repo = ChromaVectorStoreRepository(
                embeddings=self._get_embeddings().get_embeddings(),
                persist_directory=self._persist_directory
            )
        return self._vectorstore_repo
