"""Vector store repository interface."""
from abc import ABC, abstractmethod
from typing import List
from domain.entities.document import Document


class VectorStoreRepository(ABC):
    """Interface for vector store operations."""
    
    @abstractmethod
    def create_or_load(self, documents: List[Document], force_recreate: bool = False) -> None:
        """Create or load the vector store."""
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if vector store is initialized."""
        pass
