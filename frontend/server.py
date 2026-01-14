#!/usr/bin/env python3
"""
Simple HTTP server for frontend development
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8000
FRONTEND_DIR = Path(__file__).parent

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler to serve from frontend directory"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)
    
    def end_headers(self):
        # Add CORS headers for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def run_server():
    """Start the frontend server"""
    os.chdir(FRONTEND_DIR)
    
    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print(f"Frontend server running at http://localhost:{PORT}")
        print(f"Serving files from: {FRONTEND_DIR}")
        print("Press Ctrl+C to stop the server")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    run_server()
