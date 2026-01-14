"""RAG controller for handling API requests."""
import sys
from pathlib import Path
from flask import request, jsonify

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from use_cases.query_rag import QueryRAGUseCase
from use_cases.initialize_rag import InitializeRAGUseCase
from domain.entities.document import QueryResult


class RAGController:
    """Controller for RAG-related endpoints."""
    
    def __init__(
        self,
        query_use_case: QueryRAGUseCase,
        init_use_case: InitializeRAGUseCase
    ):
        """
        Initialize the controller.
        
        Args:
            query_use_case: Query RAG use case
            init_use_case: Initialize RAG use case
        """
        self.query_use_case = query_use_case
        self.init_use_case = init_use_case
    
    def query(self):
        """Handle query request."""
        try:
            data = request.get_json()
            question = data.get('question', '').strip()
            
            if not question:
                return jsonify({"error": "Question is required"}), 400
            
            # Execute use case
            result: QueryResult = self.query_use_case.execute(question)
            
            # Format response
            source_docs = []
            seen_contents = set()
            for doc in result.source_documents:
                content = doc.content[:500]  # Limit content length
                normalized = " ".join(content.lower().strip().split())
                
                # Skip duplicates
                is_duplicate = False
                for seen in seen_contents:
                    if len(normalized) > 50 and len(seen) > 50:
                        if normalized in seen or seen in normalized:
                            is_duplicate = True
                            break
                        normalized_words = set(normalized.split())
                        seen_words = set(seen.split())
                        if len(normalized_words) > 0 and len(seen_words) > 0:
                            overlap = len(normalized_words & seen_words) / max(len(normalized_words), len(seen_words))
                            if overlap > 0.8:
                                is_duplicate = True
                                break
                
                if not is_duplicate:
                    source_docs.append({
                        "content": content,
                        "metadata": doc.metadata or {}
                    })
                    seen_contents.add(normalized)
            
            return jsonify({
                "answer": result.answer,
                "sources": source_docs
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    def health(self):
        """Health check endpoint."""
        return jsonify({"status": "ok", "message": "RAG API is running"})
    
    def status(self, vectorstore_repo):
        """Get system status."""
        try:
            return jsonify({
                "initialized": vectorstore_repo.is_initialized(),
                "vectorstore_ready": vectorstore_repo.is_initialized()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
