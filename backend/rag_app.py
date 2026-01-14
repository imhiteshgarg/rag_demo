#!/usr/bin/env python3
"""
Simple RAG (Retrieval-Augmented Generation) Application
Uses open source tools: LangChain, ChromaDB, and Sentence Transformers
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Try to import PromptTemplate - location varies by LangChain version
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    try:
        from langchain.prompts import PromptTemplate
    except ImportError:
        raise ImportError("PromptTemplate not found. Please install langchain-core or langchain.")

# Try to import RetrievalQA - location varies by LangChain version
try:
    from langchain.chains import RetrievalQA
except ImportError:
    try:
        from langchain.chains.retrieval_qa import RetrievalQA
    except ImportError:
        RetrievalQA = None

# Try to import RecursiveCharacterTextSplitter - location varies by LangChain version
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        raise ImportError(
            "RecursiveCharacterTextSplitter not found. "
            "Please install langchain-text-splitters: pip install langchain-text-splitters"
        )

# Try to import Ollama - it might be in different locations depending on LangChain version
try:
    from langchain_community.llms import Ollama
except ImportError:
    try:
        from langchain.llms import Ollama
    except ImportError:
        Ollama = None


class SimpleRAG:
    """Simple RAG application using open source tools."""
    
    def __init__(self, documents_dir: str = "documents", persist_directory: str = "chroma_db"):
        """
        Initialize the RAG application.
        
        Args:
            documents_dir: Directory containing documents to index
            persist_directory: Directory to persist the vector database
        """
        self.documents_dir = documents_dir
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.qa_chain = None
        
        # Initialize embeddings using Sentence Transformers (open source)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Text splitter for chunking documents
        # Smaller chunks = more precise matches, but may miss context
        # Larger chunks = more context, but less precise matches
        # Using 500 chars with 50 overlap for better precision
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Split on paragraphs first, then sentences
        )
    
    def load_documents(self) -> List:
        """Load documents from the documents directory."""
        documents = []
        docs_path = Path(self.documents_dir)
        
        if not docs_path.exists():
            print(f"Documents directory '{self.documents_dir}' not found. Creating it...")
            docs_path.mkdir(exist_ok=True)
            return documents
        
        # Load text files
        if (docs_path / "*.txt").exists() or any(docs_path.glob("*.txt")):
            text_loader = DirectoryLoader(
                str(docs_path),
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents.extend(text_loader.load())
        
        # Load PDF files
        if (docs_path / "*.pdf").exists() or any(docs_path.glob("*.pdf")):
            pdf_loader = DirectoryLoader(
                str(docs_path),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            documents.extend(pdf_loader.load())
        
        return documents
    
    def create_vectorstore(self, force_recreate: bool = False):
        """Create or load the vector store from documents."""
        if not force_recreate and os.path.exists(self.persist_directory):
            print("Loading existing vector database...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print("Vector database loaded successfully!")
        else:
            print("Creating new vector database from documents...")
            documents = self.load_documents()
            
            if not documents:
                print(f"No documents found in '{self.documents_dir}'. Please add some documents.")
                return
            
            print(f"Loaded {len(documents)} documents")
            
            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)
            print(f"Split into {len(texts)} text chunks")
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print("Vector database created successfully!")
    
    def setup_qa_chain(self, use_ollama: bool = True, model_name: str = "llama2"):
        """
        Setup the QA chain for answering questions.
        
        Args:
            use_ollama: Whether to use Ollama (local LLM). If False, uses a simple template.
            model_name: Name of the Ollama model to use
        """
        if self.vectorstore is None:
            print("Error: Vector store not initialized. Please create it first.")
            return
        
        # Create retriever with similarity search
        # Use search_type="similarity" to ensure we're doing similarity search, not MMR
        # Increase k to 5 to get more relevant chunks, then filter/rank them
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Get more chunks for better context
        )
        print(f"Retriever configured: search_type=similarity, k=5")
        
        if use_ollama:
            if Ollama is None:
                print("Ollama not available. Install langchain-community or langchain with Ollama support.")
                use_ollama = False
            else:
                try:
                    # Try to use Ollama (local LLM)
                    llm = Ollama(model=model_name)
                    print(f"Using Ollama with model: {model_name}")
                except Exception as e:
                    print(f"Ollama not available ({e}). Using simple template-based responses.")
                    use_ollama = False
        
        if not use_ollama:
            # Fallback: Simple template-based response
            # In a real scenario, you might want to use HuggingFace transformers
            from langchain_core.language_models import BaseLLM
            from langchain_core.outputs import LLMResult, Generation
            from typing import Any, List as TypingList
            
            class SimpleLLM(BaseLLM):
                def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
                    """Generate answer based on context in the prompt."""
                    try:
                        # Extract context and question from the prompt
                        context = ""
                        question = ""
                        
                        if "Context:" in prompt and "Question:" in prompt:
                            # Split prompt into context and question sections
                            parts = prompt.split("Context:")
                            if len(parts) > 1:
                                context_part = parts[1].split("Question:")[0].strip()
                                question = parts[1].split("Question:")[1].split("Answer:")[0].strip() if "Answer:" in parts[1] else parts[1].split("Question:")[1].strip()
                                context = context_part
                        elif "Question:" in prompt:
                            # Fallback: just extract question
                            question = prompt.split("Question:")[-1].split("Answer:")[0].strip() if "Answer:" in prompt else prompt.split("Question:")[1].strip()
                            context = prompt.split("Question:")[0].replace("Use the following pieces of context to answer the question at the end.", "").replace("Context:", "").strip()
                        
                        # If we have context, try to find relevant information
                        if context and question:
                            # Simple keyword-based answer extraction
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
                                # Look for the section header first (e.g., "Common applications of AI and ML include:")
                                found_relevant_section = False
                                for i, line in enumerate(lines):
                                    line_stripped = line.strip()
                                    line_lower = line_stripped.lower()
                                    
                                    # Check if this line contains keywords from the question
                                    if any(keyword in line_lower for keyword in question_lower.split() if len(keyword) > 3):
                                        found_relevant_section = True
                                    
                                    # If we found a relevant section, collect bullet points that follow
                                    if found_relevant_section and (line_stripped.startswith('-') or line_stripped.startswith('•')):
                                        # Normalize for duplicate detection (remove bullet markers)
                                        normalized_item = line_stripped.lstrip('-•').strip().lower()
                                        if normalized_item and normalized_item not in seen_items:
                                            answer_parts.append(line_stripped)
                                            seen_items.add(normalized_item)
                                            # Stop after collecting a reasonable number (e.g., 5-10 items)
                                            if len(answer_parts) >= 10:
                                                break
                                    
                                    # If we hit a new section header, stop
                                    if found_relevant_section and line_stripped.endswith(':') and not line_stripped.startswith('-'):
                                        if i > 0:  # Don't stop on the first line
                                            break
                                
                                # If we didn't find a specific section, just get the first few bullet points (deduplicated)
                                if not answer_parts:
                                    for line in lines:
                                        line_stripped = line.strip()
                                        if line_stripped.startswith('-') or line_stripped.startswith('•'):
                                            normalized_item = line_stripped.lstrip('-•').strip().lower()
                                            if normalized_item and normalized_item not in seen_items:
                                                answer_parts.append(line_stripped)
                                                seen_items.add(normalized_item)
                                                if len(answer_parts) >= 5:  # Limit to 5 items
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
                                # Remove any duplicates that might have slipped through
                                unique_parts = []
                                seen_answers = set()
                                for part in answer_parts:
                                    normalized = " ".join(part.lower().split())
                                    if normalized not in seen_answers:
                                        unique_parts.append(part)
                                        seen_answers.add(normalized)
                                
                                answer = "\n".join(unique_parts)
                                return answer + "\n\n(Note: This is a simple template-based response. For better answers, install Ollama: 'pip install ollama', then run 'ollama pull llama2')"
                        
                        # Fallback if we can't parse
                        return "I found relevant context, but couldn't extract a specific answer. Please install Ollama for better responses: 'pip install ollama', then run 'ollama pull llama2'"
                    except Exception as e:
                        return f"Error processing prompt: {str(e)}\n\n(Note: Install Ollama for better responses: 'pip install ollama', then run 'ollama pull llama2')"
                
                def _generate(
                    self,
                    prompts: TypingList[str],
                    stop: Optional[List[str]] = None,
                    **kwargs: Any,
                ) -> LLMResult:
                    """Generate responses for the given prompts."""
                    generations = []
                    for prompt in prompts:
                        text = self._call(prompt, stop=stop, **kwargs)
                        generations.append([Generation(text=text)])
                    return LLMResult(generations=generations)
                
                @property
                def _llm_type(self) -> str:
                    return "simple"
                
                @property
                def _identifying_params(self):
                    return {}
            
            llm = SimpleLLM()
        
        # Create prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {question}

        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain - try different approaches based on LangChain version
        if RetrievalQA is not None:
            try:
                # Try the old API
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": PROMPT}
                )
            except Exception as e:
                print(f"Error creating RetrievalQA chain: {e}")
                print("Trying alternative approach...")
                self._setup_qa_chain_alternative(llm, retriever, PROMPT)
        else:
            # Use alternative approach
            self._setup_qa_chain_alternative(llm, retriever, PROMPT)
        
        print("QA chain setup complete!")
    
    def _setup_qa_chain_alternative(self, llm, retriever, prompt_template):
        """Alternative QA chain setup using LCEL (LangChain Expression Language) for LangChain 1.x"""
        from langchain_core.runnables import RunnablePassthrough, RunnableLambda
        from langchain_core.output_parsers import StrOutputParser
        
        def format_docs(docs):
            """Format documents for the prompt, removing duplicates."""
            try:
                if not docs:
                    return "No relevant documents found."
                # Ensure docs is iterable - handle different types
                if isinstance(docs, str):
                    return docs
                # Check if it's a method (shouldn't happen, but safety check)
                if callable(docs) and not isinstance(docs, (list, tuple, dict)):
                    print(f"Warning: format_docs received a callable: {type(docs)}")
                    return "Error: received callable instead of documents"
                if not isinstance(docs, (list, tuple)):
                    # If it's a single document, wrap it
                    if hasattr(docs, 'page_content'):
                        return docs.page_content
                    docs = [docs] if docs else []
                
                # Extract page_content from each document and deduplicate
                seen_contents = set()
                formatted = []
                for doc in docs:
                    if hasattr(doc, 'page_content'):
                        content = doc.page_content.strip()
                    elif isinstance(doc, str):
                        content = doc.strip()
                    else:
                        content = str(doc).strip()
                    
                    # Skip empty content
                    if not content:
                        continue
                    
                    # Create a normalized version for comparison (lowercase, no extra whitespace)
                    normalized = " ".join(content.lower().split())
                    
                    # Skip if we've seen very similar content (80% similarity threshold)
                    is_duplicate = False
                    for seen in seen_contents:
                        # Simple similarity check: if one contains the other (with some threshold)
                        if len(normalized) > 50 and len(seen) > 50:
                            # Check if one is a substantial substring of the other
                            if normalized in seen or seen in normalized:
                                is_duplicate = True
                                break
                            # Check for high overlap (simple word-based)
                            normalized_words = set(normalized.split())
                            seen_words = set(seen.split())
                            if len(normalized_words) > 0 and len(seen_words) > 0:
                                overlap = len(normalized_words & seen_words) / max(len(normalized_words), len(seen_words))
                                if overlap > 0.8:  # 80% word overlap
                                    is_duplicate = True
                                    break
                    
                    if not is_duplicate:
                        formatted.append(content)
                        seen_contents.add(normalized)
                
                return "\n\n".join(formatted)
            except Exception as e:
                print(f"Error formatting docs: {e}, type: {type(docs)}")
                import traceback
                print(traceback.format_exc())
                return "Error formatting documents."
        
        # Store retriever for source documents
        self.retriever = retriever
        
        # Create a simple chain using LCEL with proper structure
        # Use RunnableLambda to properly handle the retrieval
        def retrieve_context(input_dict):
            """Retrieve and format context documents."""
            try:
                # Extract question from input
                if isinstance(input_dict, dict):
                    question = input_dict.get("question", "")
                else:
                    question = str(input_dict)
                
                if not question:
                    return "No question provided."
                
                # Invoke retriever to get documents
                print(f"Retrieving documents for question: {question[:50]}...")
                docs = retriever.invoke(question)
                doc_count = len(docs) if isinstance(docs, list) else 1 if docs else 0
                print(f"Retrieved {doc_count} document(s)")
                
                # Debug: Print first 200 chars of each retrieved document
                if docs:
                    for i, doc in enumerate(docs[:3]):  # Show first 3
                        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                        # Try to get similarity score if available
                        score_info = ""
                        if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                            score_info = f" (score: {doc.metadata['score']:.3f})"
                        print(f"  Doc {i+1}{score_info} preview: {content[:200]}...")
                
                # Also try similarity_search_with_score to see actual scores
                if hasattr(self, 'vectorstore') and self.vectorstore:
                    try:
                        scored_docs = self.vectorstore.similarity_search_with_score(question, k=3)
                        print(f"Similarity scores:")
                        for i, (doc, score) in enumerate(scored_docs):
                            print(f"  Doc {i+1}: score={score:.4f}")
                    except Exception as e:
                        print(f"Could not get similarity scores: {e}")
                
                if not docs:
                    print("WARNING: No documents retrieved! Vectorstore may be empty or need recreation.")
                    # Try direct vectorstore search as fallback
                    if hasattr(self, 'vectorstore') and self.vectorstore:
                        print("Trying direct vectorstore search...")
                        docs = self.vectorstore.similarity_search(question, k=3)
                        print(f"Direct search found {len(docs)} document(s)")
                return format_docs(docs)
            except Exception as e:
                print(f"Error in retrieve_context: {e}")
                import traceback
                print(traceback.format_exc())
                return f"Error retrieving context: {str(e)}"
        
        # Create chain with proper LCEL structure
        # Use RunnablePassthrough.assign to add context while keeping question
        rag_chain = (
            RunnablePassthrough.assign(context=RunnableLambda(retrieve_context))
            | prompt_template
            | llm
            | StrOutputParser()
        )
        
        self.qa_chain = rag_chain
    
    def _deduplicate_source_documents(self, docs):
        """Remove duplicate source documents based on content similarity."""
        if not docs:
            return []
        
        seen_contents = set()
        unique_docs = []
        
        for doc in docs:
            if hasattr(doc, 'page_content'):
                content = doc.page_content.strip()
            elif isinstance(doc, str):
                content = doc.strip()
            else:
                content = str(doc).strip()
            
            if not content:
                continue
            
            # Create normalized version for comparison
            normalized = " ".join(content.lower().split())
            
            # Check if we've seen very similar content
            is_duplicate = False
            for seen in seen_contents:
                if len(normalized) > 50 and len(seen) > 50:
                    # Check if one is a substantial substring of the other
                    if normalized in seen or seen in normalized:
                        is_duplicate = True
                        break
                    # Check for high word overlap (80% threshold)
                    normalized_words = set(normalized.split())
                    seen_words = set(seen.split())
                    if len(normalized_words) > 0 and len(seen_words) > 0:
                        overlap = len(normalized_words & seen_words) / max(len(normalized_words), len(seen_words))
                        if overlap > 0.8:  # 80% word overlap = duplicate
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                unique_docs.append(doc)
                seen_contents.add(normalized)
        
        return unique_docs
    
    def query(self, question: str) -> dict:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary with 'result' and 'source_documents'
        """
        if self.qa_chain is None:
            return {
                "result": "Error: QA chain not initialized. Please setup the QA chain first.",
                "source_documents": []
            }
        
        # Handle different chain types
        if isinstance(self.qa_chain, dict):
            # Old RetrievalQA format
            result = self.qa_chain({"query": question})
            return result
        else:
            # New LCEL format
            try:
                # LCEL chain expects a dictionary with "question" key
                # The chain will use RunnablePassthrough.assign to add context
                answer = self.qa_chain.invoke({"question": question})
                # Get source documents using the retriever
                source_docs = []
                if hasattr(self, 'retriever') and self.retriever:
                    try:
                        # In LangChain 1.x, use invoke method
                        retrieved_docs = self.retriever.invoke(question)
                        # Ensure it's a list and contains document objects
                        if isinstance(retrieved_docs, list):
                            source_docs = retrieved_docs
                        elif retrieved_docs:
                            # If it's a single document, wrap it
                            source_docs = [retrieved_docs]
                        else:
                            source_docs = []
                    except Exception as e:
                        print(f"Error retrieving source docs: {e}")
                        # Fallback: try to get documents directly from vectorstore
                        try:
                            if hasattr(self, 'vectorstore') and self.vectorstore:
                                # Use vectorstore search directly
                                source_docs = self.vectorstore.similarity_search(question, k=3)
                        except Exception as e2:
                            print(f"Error with vectorstore search: {e2}")
                            source_docs = []
                
                # Deduplicate source documents
                if source_docs:
                    source_docs = self._deduplicate_source_documents(source_docs)
                return {
                    "result": answer if answer else "No answer generated",
                    "source_documents": source_docs
                }
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"Error in query: {e}")
                print(f"Traceback: {error_details}")
                # Don't try fallback for LCEL chains - they're not callable
                return {
                    "result": f"Error querying: {str(e)}",
                    "source_documents": []
                }
