"""Simple template-based LLM implementation."""
import sys
from pathlib import Path
from typing import Optional, List

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from domain.repositories.llm_repository import LLMRepository


class SimpleLLMRepository(LLMRepository):
    """Simple template-based LLM implementation (no real LLM)."""
    
    def generate_answer(self, context: str, question: str) -> str:
        """Generate an answer based on context and question."""
        try:
            if not context or not question:
                return "I need both context and a question to provide an answer."
            
            context_lower = context.lower()
            question_lower = question.lower()
            
            # Look for direct matches or related content
            answer_parts = []
            
            # If question asks about "applications" or "include", look for lists
            if "application" in question_lower or "include" in question_lower:
                # Find bullet points or list items in context that are relevant to the question
                lines = context.split('\n')
                # Track seen items to avoid duplicates
                seen_items = set()
                # Look for the section header first
                found_relevant_section = False
                for i, line in enumerate(lines):
                    line_stripped = line.strip()
                    line_lower = line_stripped.lower()
                    
                    # Check if this line contains keywords from the question
                    if any(keyword in line_lower for keyword in question_lower.split() if len(keyword) > 3):
                        found_relevant_section = True
                    
                    # If we found a relevant section, collect bullet points that follow
                    if found_relevant_section and (line_stripped.startswith('-') or line_stripped.startswith('•')):
                        # Normalize for duplicate detection
                        normalized_item = line_stripped.lstrip('-•').strip().lower()
                        if normalized_item and normalized_item not in seen_items:
                            answer_parts.append(line_stripped)
                            seen_items.add(normalized_item)
                            if len(answer_parts) >= 10:
                                break
                    
                    # If we hit a new section header, stop
                    if found_relevant_section and line_stripped.endswith(':') and not line_stripped.startswith('-'):
                        if i > 0:
                            break
                
                # If we didn't find a specific section, just get the first few bullet points
                if not answer_parts:
                    for line in lines:
                        line_stripped = line.strip()
                        if line_stripped.startswith('-') or line_stripped.startswith('•'):
                            normalized_item = line_stripped.lstrip('-•').strip().lower()
                            if normalized_item and normalized_item not in seen_items:
                                answer_parts.append(line_stripped)
                                seen_items.add(normalized_item)
                                if len(answer_parts) >= 5:
                                    break
            
            # If question asks "what is" or definition, find definition
            if "what is" in question_lower or "define" in question_lower:
                # Take first few sentences as definition
                sentences = context.split('.')
                if sentences:
                    answer_parts.append(sentences[0].strip() + '.')
            
            # If no specific pattern, return relevant context
            if not answer_parts:
                # Return a relevant portion of context
                if len(context) > 500:
                    answer_parts.append(context[:500] + "...")
                else:
                    answer_parts.append(context)
            
            if answer_parts:
                # Remove any duplicates
                unique_parts = []
                seen_answers = set()
                for part in answer_parts:
                    normalized = " ".join(part.lower().split())
                    if normalized not in seen_answers:
                        unique_parts.append(part)
                        seen_answers.add(normalized)
                
                answer = "\n".join(unique_parts)
                return answer + "\n\n(Note: This is a simple template-based response. For better answers, install Ollama: 'pip install ollama', then run 'ollama pull llama2')"
            
            return "I found relevant context, but couldn't extract a specific answer. Please install Ollama for better responses: 'pip install ollama', then run 'ollama pull llama2'"
        except Exception as e:
            return f"Error processing: {str(e)}\n\n(Note: Install Ollama for better responses: 'pip install ollama', then run 'ollama pull llama2')"
