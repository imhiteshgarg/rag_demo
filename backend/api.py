"""
Flask API server for RAG application
Using Clean Architecture
"""

import sys
from pathlib import Path
from flask import Flask
from flask_cors import CORS

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from application.dependencies import DIContainer

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize dependencies
BASE_DIR = Path(__file__).parent.parent
DOCUMENTS_DIR = BASE_DIR / "backend" / "documents"
VECTORSTORE_DIR = BASE_DIR / "backend" / "chroma_db"

# Ensure directories exist
DOCUMENTS_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)

# Create dependency injection container
container = DIContainer(
    documents_dir=str(DOCUMENTS_DIR),
    persist_directory=str(VECTORSTORE_DIR)
)

# Initialize on first request
_initialized = False

def ensure_initialized():
    """Ensure RAG is initialized before handling requests."""
    global _initialized
    if not _initialized:
        try:
            # Check if documents exist
            if not DOCUMENTS_DIR.exists() or not any(DOCUMENTS_DIR.glob("*.txt")):
                print(f"WARNING: No documents found in {DOCUMENTS_DIR}")
            else:
                print(f"Found documents in {DOCUMENTS_DIR}")
            
            # Check if vectorstore exists and has data
            vectorstore_exists = VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.glob("*"))
            
            # Smart vectorstore management: only recreate when necessary
            if vectorstore_exists:
                print(f"Vectorstore exists at {VECTORSTORE_DIR}")
                # Try loading it first
                container.vectorstore_repo.create_or_load([], force_recreate=False)
                # Check if vectorstore is actually populated and working
                if container.vectorstore_repo.is_initialized():
                    # Try a test query to see if it has data
                    try:
                        test_docs = container.vectorstore_repo.search("test", k=1)
                        if not test_docs:
                            print("Vectorstore appears empty, recreating...")
                            container.init_use_case.execute(force_recreate=True)
                        else:
                            print("Vectorstore loaded successfully with existing data.")
                    except Exception as e:
                        print(f"Error checking vectorstore ({e}), recreating...")
                        container.init_use_case.execute(force_recreate=True)
                else:
                    print("Vectorstore is None, recreating...")
                    container.init_use_case.execute(force_recreate=True)
            else:
                print("No vectorstore found, creating new one...")
                container.init_use_case.execute(force_recreate=True)
            
            _initialized = True
            print("RAG system initialized!")
        except Exception as e:
            print(f"Error initializing RAG: {e}")
            import traceback
            traceback.print_exc()

# Register routes
@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return container.rag_controller.health()

@app.route('/api/query', methods=['POST'])
def query():
    """Query the RAG system."""
    ensure_initialized()
    return container.rag_controller.query()

@app.route('/api/status', methods=['GET'])
def status():
    """Get the status of the RAG system."""
    try:
        ensure_initialized()
        return container.rag_controller.status(container.vectorstore_repo)
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    # Initialize before running
    ensure_initialized()
    app.run(debug=True, host='0.0.0.0', port=5000)
