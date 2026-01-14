"""
Vercel serverless function for RAG API
Using Clean Architecture
"""

import sys
import os
from pathlib import Path
from flask import Flask
from flask_cors import CORS

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, backend_path)

try:
    from application.dependencies import DIContainer
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Python path: {sys.path}")
    print(f"Backend path: {backend_path}")
    print(f"Backend path exists: {os.path.exists(backend_path)}")
    raise

app = Flask(__name__)
CORS(app)

# Initialize RAG system
BASE_DIR = Path(__file__).parent.parent
# Documents stay in repo (read-only in Vercel)
DOCUMENTS_DIR = BASE_DIR / "backend" / "documents"
# Use /tmp for vectorstore (writable in serverless)
VECTORSTORE_DIR = Path('/tmp') / "chroma_db"

DOCUMENTS_DIR.mkdir(exist_ok=True, parents=True)
VECTORSTORE_DIR.mkdir(exist_ok=True, parents=True)

# Lazy-load container to avoid OOM at import time
# Don't create container until first request
_container = None

def get_container():
    """Lazy-load container only when needed."""
    global _container
    if _container is None:
        try:
            _container = DIContainer(
                documents_dir=str(DOCUMENTS_DIR),
                persist_directory=str(VECTORSTORE_DIR)
            )
        except Exception as e:
            print(f"Error creating container: {e}")
            import traceback
            traceback.print_exc()
            raise
    return _container

_initialized = False

def ensure_initialized(container):
    global _initialized
    if not _initialized:
        try:
            # Check if vectorstore exists
            vectorstore_exists = VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.glob("*"))
            
            if vectorstore_exists:
                container.vectorstore_repo.create_or_load([], force_recreate=False)
                if container.vectorstore_repo.is_initialized():
                    test_docs = container.vectorstore_repo.search("test", k=1)
                    if not test_docs:
                        container.init_use_case.execute(force_recreate=True)
                else:
                    container.init_use_case.execute(force_recreate=True)
            else:
                container.init_use_case.execute(force_recreate=True)
            
            _initialized = True
        except Exception as e:
            print(f"Error initializing RAG: {e}")
            import traceback
            traceback.print_exc()

@app.route('/health', methods=['GET'])
def health():
    container = get_container()
    return container.rag_controller.health()

@app.route('/query', methods=['POST'])
def query():
    container = get_container()
    ensure_initialized(container)
    return container.rag_controller.query()

@app.route('/status', methods=['GET'])
def status():
    try:
        container = get_container()
        ensure_initialized(container)
        return container.rag_controller.status(container.vectorstore_repo)
    except Exception as e:
        return {"error": str(e)}, 500

# Vercel Python runtime compatibility
# The app will be used by Vercel's routing system
