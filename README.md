# PDF-ChatBot-using-Flask-Gemini-AI
This repository contains a Flask-based chatbot that enables users to upload PDFs and ask questions about their content. It leverages Google Gemini AI, FAISS vector search, and LangChain to process PDFs, extract text, create embeddings, and generate context-aware answers.

🚀 Features
✅ Upload and process PDFs to extract text
✅ Generate vector embeddings using Google Generative AI
✅ Store and retrieve document embeddings with FAISS
✅ Answer user queries based on document content
✅ API routes for file upload and question-answering
✅ Simple and interactive Flask UI

![sample](https://github.com/user-attachments/assets/cd81dc9b-c905-4d6f-adb5-f5b5b867ea6e)

🛠️ Tech Stack
Flask – Web framework
LangChain – Text processing & LLM integration
Google Gemini AI – Embeddings & Chat model
FAISS – Vector similarity search
PyPDF2 – PDF text extraction

⚡ Getting Started
Clone the repository:
git clone https://github.com/yourusername/repository-name.git
cd repository-name
Install dependencies:
pip install -r requirements.txt
Set up Google API credentials in .env:
my_key=YOUR_GOOGLE_API_KEY

Run the Flask app:
python app.py

Open your browser and navigate to:
http://127.0.0.1:5000

📌 API Endpoints
POST /upload → Upload a PDF file
POST /chat → Ask a question about the uploaded PDF
🎯 Use Cases
AI-powered document search
Automated report analysis
Legal, research, and educational applications



