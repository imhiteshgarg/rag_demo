# Design Patterns Used in RAG Application

This document outlines all the design patterns implemented in the RAG application's clean architecture.

## 1. Repository Pattern

**Purpose**: Abstracts data access logic and provides a uniform interface for data operations.

**Implementation**:
- **Interfaces**: `domain/repositories/` (VectorStoreRepository, DocumentRepository, LLMRepository)
- **Implementations**: `infrastructure/vectorstore/chroma_vectorstore.py`, `infrastructure/document_loaders/langchain_loader.py`, `infrastructure/llm/simple_llm.py`

**Example**:
```python
# Interface (domain/repositories/vectorstore_repository.py)
class VectorStoreRepository(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Document]:
        pass

# Implementation (infrastructure/vectorstore/chroma_vectorstore.py)
class ChromaVectorStoreRepository(VectorStoreRepository):
    def search(self, query: str, k: int = 5) -> List[Document]:
        # ChromaDB-specific implementation
```

**Benefits**:
- Decouples business logic from data access
- Easy to swap implementations (e.g., ChromaDB → Pinecone)
- Testable with mock repositories

---

## 2. Dependency Injection (DI) Pattern

**Purpose**: Inverts control of dependencies, making code more testable and maintainable.

**Implementation**: `application/dependencies.py` - `DIContainer` class

**Example**:
```python
class DIContainer:
    def __init__(self, documents_dir: str, persist_directory: str):
        # Create dependencies
        self._embeddings = SentenceTransformersEmbeddings()
        self._vectorstore_repo = ChromaVectorStoreRepository(...)
        
        # Inject dependencies into use cases
        self._query_use_case = QueryRAGUseCase(
            vectorstore_repo=self._vectorstore_repo,
            llm_repo=self._llm_repo
        )
```

**Benefits**:
- Loose coupling between components
- Easy to test with mock dependencies
- Centralized dependency management

---

## 3. Strategy Pattern

**Purpose**: Defines a family of algorithms, encapsulates each one, and makes them interchangeable.

**Implementation**: Different repository implementations can be swapped without changing use cases.

**Example**:
```python
# Strategy 1: ChromaDB
vectorstore_repo = ChromaVectorStoreRepository(...)

# Strategy 2: Could be Pinecone, Weaviate, etc.
# vectorstore_repo = PineconeVectorStoreRepository(...)

# Use case doesn't change
use_case = QueryRAGUseCase(vectorstore_repo=vectorstore_repo, ...)
```

**Benefits**:
- Runtime selection of algorithms
- Open/Closed Principle compliance
- Easy to add new strategies

---

## 4. Use Case Pattern (Application Service Pattern)

**Purpose**: Encapsulates application-specific business logic in a single, testable unit.

**Implementation**: `use_cases/query_rag.py`, `use_cases/initialize_rag.py`

**Example**:
```python
class QueryRAGUseCase:
    def execute(self, question: str) -> QueryResult:
        # 1. Retrieve documents
        source_docs = self.vectorstore_repo.search(question, k=5)
        
        # 2. Format context
        context = self._format_context(source_docs)
        
        # 3. Generate answer
        answer = self.llm_repo.generate_answer(context, question)
        
        return QueryResult(answer=answer, source_documents=source_docs)
```

**Benefits**:
- Single responsibility per use case
- Clear business logic flow
- Easy to test and understand

---

## 5. Controller Pattern (MVC/Adapter Pattern)

**Purpose**: Handles HTTP requests and adapts them to use cases, then formats responses.

**Implementation**: `interfaces/controllers/rag_controller.py`

**Example**:
```python
class RAGController:
    def query(self):
        # 1. Extract request data
        data = request.get_json()
        question = data.get('question', '').strip()
        
        # 2. Execute use case
        result = self.query_use_case.execute(question)
        
        # 3. Format HTTP response
        return jsonify({
            "answer": result.answer,
            "sources": source_docs
        })
```

**Benefits**:
- Separates HTTP concerns from business logic
- Easy to change API format (REST → GraphQL)
- Testable without HTTP layer

---

## 6. Factory Pattern

**Purpose**: Creates objects without specifying the exact class of object that will be created.

**Implementation**: `DIContainer` acts as a factory for creating and wiring dependencies.

**Example**:
```python
class DIContainer:
    def __init__(self, ...):
        # Factory creates all dependencies
        self._embeddings = SentenceTransformersEmbeddings()
        self._vectorstore_repo = ChromaVectorStoreRepository(...)
        # ... creates and wires everything
```

**Benefits**:
- Centralized object creation
- Hides complexity of object construction
- Easy to change implementations

---

## 7. Adapter Pattern

**Purpose**: Allows incompatible interfaces to work together.

**Implementation**: 
- `LangChainDocumentRepository` adapts LangChain's document format to our `Document` entity
- `ChromaVectorStoreRepository` adapts ChromaDB to our `VectorStoreRepository` interface

**Example**:
```python
# LangChain uses Document with page_content
langchain_doc = LangChainDocument(page_content="...", metadata={})

# We adapt it to our domain Document
domain_doc = Document(
    content=langchain_doc.page_content,
    metadata=langchain_doc.metadata
)
```

**Benefits**:
- Integrates third-party libraries cleanly
- Keeps domain model independent
- Easy to swap libraries

---

## 8. Dependency Inversion Principle (DIP)

**Purpose**: High-level modules should not depend on low-level modules; both should depend on abstractions.

**Implementation**: 
- Use cases depend on repository interfaces (abstractions)
- Infrastructure implements those interfaces (concretions)
- Domain layer has no dependencies on infrastructure

**Example**:
```python
# Use case depends on abstraction
class QueryRAGUseCase:
    def __init__(self, vectorstore_repo: VectorStoreRepository, ...):
        # VectorStoreRepository is an abstract interface
        self.vectorstore_repo = vectorstore_repo

# Infrastructure provides concrete implementation
class ChromaVectorStoreRepository(VectorStoreRepository):
    # Concrete implementation
```

**Benefits**:
- Flexible architecture
- Easy to test (mock interfaces)
- Follows SOLID principles

---

## 9. Clean Architecture / Layered Architecture

**Purpose**: Organizes code into layers with clear boundaries and dependency rules.

**Layers** (from inner to outer):
1. **Domain** (Entities, Repository Interfaces) - No dependencies
2. **Use Cases** - Depends only on domain
3. **Interfaces** (Controllers) - Depends on use cases
4. **Infrastructure** - Implements domain interfaces
5. **Application** (DI Container) - Wires everything together

**Benefits**:
- Clear separation of concerns
- Independent of frameworks
- Testable at each layer
- Easy to maintain and extend

---

## 10. Template Method Pattern (Partial)

**Purpose**: Defines the skeleton of an algorithm, deferring some steps to subclasses.

**Implementation**: Use cases define the flow, but specific steps can vary.

**Example**:
```python
class QueryRAGUseCase:
    def execute(self, question: str) -> QueryResult:
        # Template: fixed algorithm steps
        source_docs = self.vectorstore_repo.search(...)  # Step 1
        context = self._format_context(source_docs)       # Step 2
        answer = self.llm_repo.generate_answer(...)       # Step 3
        return QueryResult(...)                           # Step 4
```

**Benefits**:
- Consistent algorithm structure
- Reusable code
- Easy to extend

---

## 11. Singleton Pattern (Implicit)

**Purpose**: Ensures a class has only one instance and provides global access.

**Implementation**: `DIContainer` instance in `api.py` (one instance per application)

**Example**:
```python
# In api.py
container = DIContainer(...)  # Single instance

@app.route('/api/query')
def query():
    return container.rag_controller.query()  # Uses singleton
```

**Benefits**:
- Single source of truth for dependencies
- Efficient resource usage
- Centralized configuration

---

## Summary Table

| Pattern | Location | Purpose |
|---------|----------|---------|
| Repository | `domain/repositories/` | Abstract data access |
| Dependency Injection | `application/dependencies.py` | Manage dependencies |
| Strategy | Repository implementations | Interchangeable algorithms |
| Use Case | `use_cases/` | Business logic encapsulation |
| Controller | `interfaces/controllers/` | HTTP request handling |
| Factory | `DIContainer` | Object creation |
| Adapter | Infrastructure implementations | Interface compatibility |
| Dependency Inversion | Overall architecture | Abstractions over concretions |
| Clean Architecture | Entire structure | Layered organization |
| Template Method | Use cases | Algorithm structure |
| Singleton | `DIContainer` instance | Single instance management |

---

## Design Principles Applied

1. **SOLID Principles**:
   - **S**ingle Responsibility: Each class has one reason to change
   - **O**pen/Closed: Open for extension, closed for modification
   - **L**iskov Substitution: Repository implementations are substitutable
   - **I**nterface Segregation: Focused repository interfaces
   - **D**ependency Inversion: Depend on abstractions

2. **DRY (Don't Repeat Yourself)**: Shared logic in use cases and utilities

3. **Separation of Concerns**: Clear boundaries between layers

4. **Testability**: All patterns support easy unit testing
