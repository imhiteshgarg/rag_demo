# Clean Architecture - Backend Structure

This backend follows **Clean Architecture** principles with clear separation of concerns.

## Directory Structure

```
backend/
├── domain/                    # Core business logic (innermost layer)
│   ├── entities/             # Domain entities (Document, QueryResult)
│   │   └── document.py
│   └── repositories/         # Repository interfaces (abstractions)
│       ├── vectorstore_repository.py
│       ├── document_repository.py
│       └── llm_repository.py
│
├── use_cases/                 # Application business logic
│   ├── query_rag.py          # Query documents use case
│   └── initialize_rag.py     # Initialize RAG system use case
│
├── interfaces/                # Interface adapters
│   └── controllers/          # HTTP controllers
│       └── rag_controller.py
│
├── infrastructure/            # External services (outermost layer)
│   ├── document_loaders/      # Document loading implementations
│   │   └── langchain_loader.py
│   ├── embeddings/            # Embedding implementations
│   │   └── sentence_transformers.py
│   ├── vectorstore/           # Vector store implementations
│   │   └── chroma_vectorstore.py
│   └── llm/                   # LLM implementations
│       └── simple_llm.py
│
├── application/               # Dependency injection & configuration
│   └── dependencies.py        # DI Container
│
├── api.py                     # Flask application entry point
├── rag_app.py                 # Legacy file (can be removed)
└── documents/                 # Document storage
```

## Architecture Layers

### 1. Domain Layer (Core)
- **Entities**: Core business objects (`Document`, `QueryResult`)
- **Repository Interfaces**: Abstract contracts for data access
- **No dependencies** on other layers

### 2. Use Cases Layer
- **Business Logic**: Application-specific workflows
- **Depends on**: Domain layer only
- Examples: `QueryRAGUseCase`, `InitializeRAGUseCase`

### 3. Interface Adapters Layer
- **Controllers**: Handle HTTP requests/responses
- **Depends on**: Use cases and domain
- Example: `RAGController`

### 4. Infrastructure Layer
- **Implementations**: Concrete implementations of repository interfaces
- **External Services**: LangChain, ChromaDB, Sentence Transformers
- **Depends on**: Domain interfaces

### 5. Application Layer
- **Dependency Injection**: Wires everything together
- **Configuration**: Sets up dependencies
- **Depends on**: All layers

## Dependency Rule

**Dependencies point inward:**
- Infrastructure → Domain (implements interfaces)
- Use Cases → Domain (uses entities and interfaces)
- Controllers → Use Cases → Domain
- Application → All layers (wires them together)

## Benefits

1. **Testability**: Easy to mock repositories for testing
2. **Flexibility**: Swap implementations (e.g., different vector stores)
3. **Maintainability**: Clear separation of concerns
4. **Scalability**: Easy to add new features without breaking existing code

## Usage

The `DIContainer` in `application/dependencies.py` wires all dependencies together. The API (`api.py`) uses the container to get controllers and use cases.

## Design Patterns

See [DESIGN_PATTERNS.md](./DESIGN_PATTERNS.md) for a detailed explanation of all design patterns used in this application, including:
- Repository Pattern
- Dependency Injection
- Strategy Pattern
- Use Case Pattern
- Controller Pattern
- Factory Pattern
- Adapter Pattern
- And more...

## Migration Notes

- `rag_app.py` is kept for reference but is no longer used
- All functionality has been moved to the clean architecture structure
- The API interface remains the same, so no frontend changes needed
