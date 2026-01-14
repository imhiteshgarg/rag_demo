"""Use case for initializing the RAG system."""
import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from domain.repositories.vectorstore_repository import VectorStoreRepository
from domain.repositories.document_repository import DocumentRepository


class InitializeRAGUseCase:
    """Use case for initializing the RAG system."""
    
    def __init__(
        self,
        document_repo: DocumentRepository,
        vectorstore_repo: VectorStoreRepository,
        documents_dir: str
    ):
        """
        Initialize the use case.
        
        Args:
            document_repo: Document repository
            vectorstore_repo: Vector store repository
            documents_dir: Directory containing documents
        """
        self.document_repo = document_repo
        self.vectorstore_repo = vectorstore_repo
        self.documents_dir = documents_dir
    
    def execute(self, force_recreate: bool = False) -> bool:
        """
        Execute the initialization use case.
        
        Args:
            force_recreate: Whether to force recreation of vector store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load documents
            documents = self.document_repo.load_documents(self.documents_dir)
            
            if not documents:
                print(f"No documents found in '{self.documents_dir}'")
                return False
            
            # Create or load vector store
            self.vectorstore_repo.create_or_load(documents, force_recreate=force_recreate)
            
            return True
        except Exception as e:
            print(f"Error initializing RAG: {e}")
            return False
