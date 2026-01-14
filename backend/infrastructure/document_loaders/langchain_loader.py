"""LangChain-based document loader implementation."""
import sys
from pathlib import Path
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from domain.entities.document import Document
from domain.repositories.document_repository import DocumentRepository


class LangChainDocumentRepository(DocumentRepository):
    """LangChain-based document repository implementation."""
    
    def load_documents(self, documents_dir: str) -> List[Document]:
        """Load documents from a directory."""
        documents = []
        docs_path = Path(documents_dir)
        
        if not docs_path.exists():
            print(f"Documents directory '{documents_dir}' not found. Creating it...")
            docs_path.mkdir(exist_ok=True)
            return documents
        
        # Load text files
        if any(docs_path.glob("*.txt")):
            text_loader = DirectoryLoader(
                str(docs_path),
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            langchain_docs = text_loader.load()
            for doc in langchain_docs:
                documents.append(Document(
                    content=doc.page_content,
                    metadata=doc.metadata if hasattr(doc, 'metadata') else {}
                ))
        
        # Load PDF files
        if any(docs_path.glob("*.pdf")):
            pdf_loader = DirectoryLoader(
                str(docs_path),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            langchain_docs = pdf_loader.load()
            for doc in langchain_docs:
                documents.append(Document(
                    content=doc.page_content,
                    metadata=doc.metadata if hasattr(doc, 'metadata') else {}
                ))
        
        return documents
