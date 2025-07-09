# frontend/server.py
import http.server
import socketserver
import os

PORT = 8000 # The port our service will run on

# Ensure the server runs from the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving simple HTTP frontend on port {PORT}")
    httpd.serve_forever()