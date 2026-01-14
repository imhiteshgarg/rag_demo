"""Use case for querying the RAG system."""
import sys
from pathlib import Path
from typing import List

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from domain.entities.document import QueryResult, Document
from domain.repositories.vectorstore_repository import VectorStoreRepository
from domain.repositories.llm_repository import LLMRepository


class QueryRAGUseCase:
    """Use case for querying documents using RAG."""
    
    def __init__(
        self,
        vectorstore_repo: VectorStoreRepository,
        llm_repo: LLMRepository,
        max_sources: int = 5
    ):
        """
        Initialize the use case.
        
        Args:
            vectorstore_repo: Vector store repository
            llm_repo: LLM repository
            max_sources: Maximum number of source documents to retrieve
        """
        self.vectorstore_repo = vectorstore_repo
        self.llm_repo = llm_repo
        self.max_sources = max_sources
    
    def execute(self, question: str) -> QueryResult:
        """
        Execute the query use case.
        
        Args:
            question: The question to ask
            
        Returns:
            QueryResult with answer and source documents
        """
        if not question or not question.strip():
            return QueryResult(
                answer="Please provide a valid question.",
                source_documents=[]
            )
        
        # Retrieve relevant documents
        source_docs = self.vectorstore_repo.search(question, k=self.max_sources)
        
        # Format context from source documents
        context = self._format_context(source_docs)
        
        # Generate answer using LLM
        answer = self.llm_repo.generate_answer(context, question)
        
        return QueryResult(
            answer=answer,
            source_documents=source_docs
        )
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format documents into context string."""
        if not documents:
            return "No relevant documents found."
        
        # Deduplicate documents
        unique_docs = self._deduplicate_documents(documents)
        
        # Format as context
        context_parts = []
        for doc in unique_docs:
            context_parts.append(doc.content)
        
        return "\n\n".join(context_parts)
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content similarity."""
        if not documents:
            return []
        
        seen_contents = set()
        unique_docs = []
        
        for doc in documents:
            content = doc.content.strip()
            if not content:
                continue
            
            # Normalize for comparison
            normalized = " ".join(content.lower().split())
            
            # Check for duplicates
            is_duplicate = False
            for seen in seen_contents:
                if len(normalized) > 50 and len(seen) > 50:
                    # Check substring match
                    if normalized in seen or seen in normalized:
                        is_duplicate = True
                        break
                    # Check word overlap (80% threshold)
                    normalized_words = set(normalized.split())
                    seen_words = set(seen.split())
                    if len(normalized_words) > 0 and len(seen_words) > 0:
                        overlap = len(normalized_words & seen_words) / max(len(normalized_words), len(seen_words))
                        if overlap > 0.8:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                unique_docs.append(doc)
                seen_contents.add(normalized)
        
        return unique_docs
