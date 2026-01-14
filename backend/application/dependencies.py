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
        
        # Initialize infrastructure
        self._embeddings = SentenceTransformersEmbeddings()
        self._document_repo = LangChainDocumentRepository()
        self._vectorstore_repo = ChromaVectorStoreRepository(
            embeddings=self._embeddings.get_embeddings(),
            persist_directory=persist_directory
        )
        self._llm_repo = SimpleLLMRepository()
        
        # Initialize use cases
        self._init_use_case = InitializeRAGUseCase(
            document_repo=self._document_repo,
            vectorstore_repo=self._vectorstore_repo,
            documents_dir=documents_dir
        )
        self._query_use_case = QueryRAGUseCase(
            vectorstore_repo=self._vectorstore_repo,
            llm_repo=self._llm_repo
        )
        
        # Initialize controller
        self._rag_controller = RAGController(
            query_use_case=self._query_use_case,
            init_use_case=self._init_use_case
        )
    
    @property
    def rag_controller(self) -> RAGController:
        """Get RAG controller."""
        return self._rag_controller
    
    @property
    def init_use_case(self) -> InitializeRAGUseCase:
        """Get initialization use case."""
        return self._init_use_case
    
    @property
    def vectorstore_repo(self):
        """Get vectorstore repository."""
        return self._vectorstore_repo
