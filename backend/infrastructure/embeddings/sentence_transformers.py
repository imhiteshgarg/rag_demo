"""Sentence Transformers embeddings implementation."""
from langchain_community.embeddings import HuggingFaceEmbeddings


class SentenceTransformersEmbeddings:
    """Sentence Transformers embeddings wrapper."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embeddings.
        
        Args:
            model_name: Name of the Sentence Transformers model
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    def get_embeddings(self):
        """Get the embeddings object."""
        return self.embeddings
