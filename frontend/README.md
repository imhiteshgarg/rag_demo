# Frontend - RAG Application

Modern web interface for the RAG application.

## Files

- `index.html` - Main HTML structure
- `styles.css` - Styling and layout
- `script.js` - Frontend logic and API communication

## Local Development

### Option 1: Using the server script (Recommended)
```bash
cd frontend
python server.py
```

The server will start on `http://localhost:8000`

### Option 2: Using Python's built-in server
```bash
cd frontend
python -m http.server 8000
```

Then navigate to `http://localhost:8000` in your browser.

### Option 3: Open directly
Simply open `index.html` in your browser (may have CORS issues with API calls)

## Configuration

Update `API_BASE_URL` in `script.js` to match your backend URL:
- Local: `http://localhost:5000/api`
- Production: `/api`
