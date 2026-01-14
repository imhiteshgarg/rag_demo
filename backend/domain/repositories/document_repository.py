"""Document repository interface."""
from abc import ABC, abstractmethod
from typing import List
from domain.entities.document import Document


class DocumentRepository(ABC):
    """Interface for document loading operations."""
    
    @abstractmethod
    def load_documents(self, documents_dir: str) -> List[Document]:
        """Load documents from a directory."""
        pass
