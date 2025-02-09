from flask import Flask, request, jsonify, render_template
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from google.oauth2 import service_account

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load credentials and API keys
SERVICE_ACCOUNT_FILE = "/media/imran/Office1/SkyOps AI Works/ConverstionalBot/fluid-mix-425605-u0-b749366e3d3e.json"  # Update this
api_key = os.getenv("my_key")
scoped_credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=["https://www.googleapis.com/auth/generative-language"]
)

# Initialize global variables
vector_store = None

# PDF processing function
def process_pdf(pdf_path):
    try:
        # Read and extract text from the PDF
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text_chunks = text_splitter.split_text(text)

        # Create embeddings and FAISS vector store
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
            credentials=scoped_credentials
        )
        global vector_store  # Update global vector store
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

        return "Processing complete!"
    except Exception as e:
        return f"Error processing PDF: {e}"

# Chat function using FAISS and LLM
def generate_answer(question):
    try:
        if not vector_store:
            return "No PDF has been processed yet. Please upload a PDF first."

        # Perform similarity search
        docs = vector_store.similarity_search(question)
        if not docs:
            return "No relevant information in the context."

        # Define the prompt template
        prompt_template = """
        Answer the question based on the provided context. If no answer is available, respond: "No relevant information in the context."
        Context:
        {context}
        Question:
        {question}
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        # Generate an answer
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        return f"Error generating answer: {e}"

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    pdf = request.files.get("pdf")
    if not pdf:
        return jsonify({"error": "No file uploaded"}), 400

    # Save the file to the server
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf.filename)
    pdf.save(pdf_path)

    # Process the PDF for embeddings
    processing_status = process_pdf(pdf_path)

    # Return the URL of the uploaded PDF file for frontend to access
    pdf_url = f"/static/uploads/{pdf.filename}"

    return jsonify({"status": processing_status, "file_url": pdf_url})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Generate an answer using LLMa
    answer = generate_answer(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
