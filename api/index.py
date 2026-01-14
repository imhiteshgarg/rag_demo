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

from application.dependencies import DIContainer

app = Flask(__name__)
CORS(app)

# Initialize RAG system
BASE_DIR = Path(__file__).parent.parent
DOCUMENTS_DIR = BASE_DIR / "backend" / "documents"
VECTORSTORE_DIR = BASE_DIR / "backend" / "chroma_db"

DOCUMENTS_DIR.mkdir(exist_ok=True, parents=True)
VECTORSTORE_DIR.mkdir(exist_ok=True, parents=True)

# Create dependency injection container
container = DIContainer(
    documents_dir=str(DOCUMENTS_DIR),
    persist_directory=str(VECTORSTORE_DIR)
)

_initialized = False

def ensure_initialized():
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

@app.route('/api/health', methods=['GET'])
def health():
    return container.rag_controller.health()

@app.route('/api/query', methods=['POST'])
def query():
    ensure_initialized()
    return container.rag_controller.query()

@app.route('/api/status', methods=['GET'])
def status():
    try:
        ensure_initialized()
        return container.rag_controller.status(container.vectorstore_repo)
    except Exception as e:
        return {"error": str(e)}, 500

# Vercel Python runtime compatibility
# The app will be used by Vercel's routing system
