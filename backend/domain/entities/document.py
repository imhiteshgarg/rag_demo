"""Document entity."""
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Document:
    """Represents a document in the system."""
    content: str
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QueryResult:
    """Represents the result of a query."""
    answer: str
    source_documents: list[Document]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "result": self.answer,
            "source_documents": [
                {
                    "content": doc.content,
                    "metadata": doc.metadata or {}
                }
                for doc in self.source_documents
            ]
        }
