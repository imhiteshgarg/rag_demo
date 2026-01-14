"""LLM repository interface."""
from abc import ABC, abstractmethod


class LLMRepository(ABC):
    """Interface for LLM operations."""
    
    @abstractmethod
    def generate_answer(self, context: str, question: str) -> str:
        """Generate an answer based on context and question."""
        pass
