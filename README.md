# RAG Application

A simple Retrieval-Augmented Generation (RAG) application built with open source tools, featuring a web frontend and REST API backend.

## Project Structure

```
RAG/
├── frontend/              # Frontend web application
│   ├── index.html        # Main HTML file
│   ├── styles.css        # Styling
│   └── script.js         # Frontend JavaScript
├── backend/              # Backend Python application
│   ├── rag_app.py        # RAG core logic
│   ├── api.py            # Flask API server
│   ├── requirements.txt  # Python dependencies
│   └── documents/        # Document storage
├── api/                  # Vercel serverless function
│   └── index.py          # Serverless API handler
├── vercel.json           # Vercel configuration
└── README.md             # This file
```

## Features

- **Document Loading**: Supports text files (.txt) and PDF files (.pdf)
- **Vector Storage**: Uses ChromaDB for efficient similarity search
- **Embeddings**: Uses Sentence Transformers (all-MiniLM-L6-v2) for generating embeddings
- **Web Interface**: Modern, responsive web UI
- **REST API**: Flask-based API for querying documents

## Local Development

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. **Install Backend Dependencies**:
```bash
cd backend
pip install -r requirements.txt
```

2. **Run the Backend API**:
```bash
cd backend
python api.py
```

The API will run on `http://localhost:5000`

3. **Start the Frontend Server**:
```bash
cd frontend
python server.py
```

The frontend will be available at `http://localhost:8000`

**Note**: Make sure the backend API is running on port 5000 for the frontend to work properly.

## Deployment to Vercel

### Option 1: Using Vercel CLI

1. **Install Vercel CLI**:
```bash
npm i -g vercel
```

2. **Deploy**:
```bash
vercel
```

3. **Follow the prompts** to configure your deployment.

### Option 2: Using Vercel Dashboard

1. Push your code to GitHub
2. Import the repository in Vercel
3. Vercel will automatically detect the configuration from `vercel.json`

### Important Notes for Vercel Deployment

- **Vector Database**: ChromaDB files are stored in `/tmp` in serverless environments (they may not persist between invocations)
- **Document Storage**: For production, consider using cloud storage (S3, etc.) for documents
- **Model Size**: Sentence Transformers models are downloaded on first use - this may cause cold start delays
- **Memory Limits**: Vercel has memory limits - large documents may need chunking optimization

## API Endpoints

### Health Check
```
GET /api/health
```

### Query Documents
```
POST /api/query
Content-Type: application/json

{
  "question": "What is RAG?"
}
```

Response:
```json
{
  "answer": "RAG stands for...",
  "sources": [
    {
      "content": "Document content...",
      "metadata": {}
    }
  ]
}
```

### Status
```
GET /api/status
```

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python, Flask
- **RAG Framework**: LangChain
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers
- **Deployment**: Vercel

## Adding Documents

Place your documents (`.txt` or `.pdf` files) in the `backend/documents/` directory. The vector database will be automatically created on first run.

## Troubleshooting

### API not responding
- Check that the backend is running on port 5000
- Verify the API_BASE_URL in `frontend/script.js` matches your setup

### Vector database issues
- Delete `backend/chroma_db/` and restart to recreate
- Ensure documents are in `backend/documents/`

### Vercel deployment issues
- Check Vercel function logs for errors
- Ensure all dependencies are in `backend/requirements.txt`
- Verify `vercel.json` configuration

## License

This project uses open source tools and is provided as-is for educational purposes.
