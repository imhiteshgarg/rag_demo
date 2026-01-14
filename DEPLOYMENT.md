# Deployment Guide

## Project Structure

```
RAG/
├── frontend/          # Frontend web files (HTML, CSS, JS)
├── backend/           # Backend Python code
│   ├── rag_app.py     # RAG core logic
│   ├── api.py         # Flask API (for local dev)
│   └── documents/     # Document storage
├── api/               # Vercel serverless function
│   └── index.py       # Serverless API handler
└── vercel.json        # Vercel configuration
```

## Local Development

### Backend
```bash
cd backend
pip install -r requirements.txt
python api.py
```

### Frontend
Open `frontend/index.html` in a browser, or use:
```bash
cd frontend
python -m http.server 8000
```

## Vercel Deployment

### Prerequisites
- Vercel account
- Vercel CLI (optional): `npm i -g vercel`

### Steps

1. **Push to GitHub** (recommended)
   - Create a GitHub repository
   - Push your code

2. **Deploy via Vercel Dashboard**
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will auto-detect the configuration

3. **Or Deploy via CLI**
   ```bash
   vercel
   ```

### Important Notes

- **Documents**: Place documents in `backend/documents/` before deployment
- **Vector Database**: ChromaDB will be created automatically on first request
- **Cold Starts**: First request may be slow due to model downloads
- **Memory**: Vercel has memory limits - large documents may need optimization
- **Persistence**: Vector database in `/tmp` may not persist between deployments

### Environment Variables

No environment variables required for basic setup.

### Custom Domain

After deployment, you can add a custom domain in Vercel dashboard.

## Troubleshooting

### API not working
- Check Vercel function logs
- Verify `api/requirements.txt` has all dependencies
- Ensure documents are in `backend/documents/`

### Frontend can't connect to API
- Update `API_BASE_URL` in `frontend/script.js`
- Check CORS settings in `api/index.py`

### Vector database issues
- Documents must be committed to the repository
- Check file paths in `api/index.py`
