// API base URL - adjust for your deployment
// For local development: 'http://localhost:5000/api'
// For Vercel: '/api'
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:5000/api' 
    : '/api';

// Get DOM elements
const questionInput = document.getElementById('questionInput');
const submitBtn = document.getElementById('submitBtn');
const btnText = document.getElementById('btnText');
const btnLoader = document.getElementById('btnLoader');
const answerSection = document.getElementById('answerSection');
const answerContent = document.getElementById('answerContent');
const sourcesSection = document.getElementById('sourcesSection');
const sourcesContent = document.getElementById('sourcesContent');
const errorSection = document.getElementById('errorSection');
const errorContent = document.getElementById('errorContent');

// Handle Enter key (Shift+Enter for new line)
questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitQuery();
    }
});

async function submitQuery() {
    const question = questionInput.value.trim();
    
    if (!question) {
        showError('Please enter a question');
        return;
    }
    
    // Hide previous results
    hideAllSections();
    
    // Show loading state
    setLoading(true);
    
    try {
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to get answer');
        }
        
        // Display answer
        if (data.answer) {
            answerContent.textContent = data.answer;
            answerSection.style.display = 'block';
        }
        
        // Display sources
        if (data.sources && data.sources.length > 0) {
            displaySources(data.sources);
            sourcesSection.style.display = 'block';
        }
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'An error occurred while processing your question');
    } finally {
        setLoading(false);
    }
}

function displaySources(sources) {
    sourcesContent.innerHTML = '';
    
    sources.forEach((source, index) => {
        const sourceItem = document.createElement('div');
        sourceItem.className = 'source-item';
        
        const content = document.createElement('p');
        content.textContent = source.content || 'No content available';
        
        const meta = document.createElement('div');
        meta.className = 'source-meta';
        meta.textContent = `Source ${index + 1}${source.metadata && source.metadata.source ? ` - ${source.metadata.source}` : ''}`;
        
        sourceItem.appendChild(content);
        sourceItem.appendChild(meta);
        sourcesContent.appendChild(sourceItem);
    });
}

function showError(message) {
    errorContent.textContent = message;
    errorSection.style.display = 'block';
}

function hideAllSections() {
    answerSection.style.display = 'none';
    sourcesSection.style.display = 'none';
    errorSection.style.display = 'none';
}

function setLoading(loading) {
    submitBtn.disabled = loading;
    btnText.style.display = loading ? 'none' : 'inline';
    btnLoader.style.display = loading ? 'inline-block' : 'none';
}

// Check API health on load
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) {
            console.warn('API health check failed');
        }
    } catch (error) {
        console.warn('API health check error:', error);
    }
}

// Check health when page loads
checkHealth();
