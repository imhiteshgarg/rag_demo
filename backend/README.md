# Backend - RAG Application (Clean Architecture)

Backend API server for the RAG application, following Clean Architecture principles.

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed architecture documentation.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add documents to the `documents/` folder

3. Run the API server:
```bash
python api.py
```

The API will be available at `http://localhost:5000`

## Project Structure

```
backend/
├── domain/              # Core business logic
│   ├── entities/        # Domain entities
│   └── repositories/    # Repository interfaces
├── use_cases/           # Application business logic
├── interfaces/          # Interface adapters (controllers)
├── infrastructure/      # External services implementations
├── application/         # Dependency injection
└── api.py              # Flask application entry point
```

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/query` - Query the RAG system
- `GET /api/status` - Get system status

## Configuration

- Documents directory: `documents/`
- Vector database: `chroma_db/`
- Port: 5000 (configurable in `api.py`)

## Note

- `rag_app.py` is kept for reference but is deprecated
- All functionality has been migrated to the clean architecture structure
