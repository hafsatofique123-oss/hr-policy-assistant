
# 📘 HR Policy Assistant

An AI-powered HR Policy Assistant built using Python, Streamlit, Groq API, and Retrieval-Augmented Generation (RAG).

This application allows users to upload HR policy PDF documents and ask questions about company policies. The application retrieves relevant information from the uploaded document and generates answers using the Groq LLM.

## ✨ Features

- Upload HR policy PDF documents.
- Extract text from PDF files.
- Split documents into smaller chunks.
- Generate embeddings using Sentence Transformers.
- Store and search embeddings using FAISS.
- Ask questions about uploaded HR policies.
- Generate AI-powered answers using Groq.
- Display retrieved policy sources and PDF page numbers.
- User-friendly Streamlit interface.

## 🛠️ Technologies Used

- Python
- Streamlit
- Groq API
- openai/gpt-oss-120b
- PyMuPDF
- LangChain Text Splitters
- Sentence Transformers
- FAISS
- NumPy

## 📂 Project Structure

```text
hr-policy-assistant/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
│
└── data/
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/hr-policy-assistant.git
```

### 2. Navigate to the project folder

```bash
cd hr-policy-assistant
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 🔑 Groq API Key

1. Create a Groq API key from the Groq Console.
2. Run the application.
3. Enter your API key in the Streamlit sidebar.

Never upload your API key to GitHub or share it publicly.

## ▶️ Run the Application

```bash
streamlit run app.py
```

## 📖 How to Use

1. Open the Streamlit application.
2. Upload an HR policy PDF.
3. Click the "Process Policy" button.
4. Wait for the document to be processed.
5. Ask questions about the uploaded policy.
6. Read the AI-generated answer and retrieved sources.

## 🧠 How RAG Works

1. The application extracts text from the uploaded PDF.
2. The text is divided into smaller chunks.
3. Embeddings are created for each chunk.
4. Embeddings are stored in a FAISS vector index.
5. Relevant chunks are retrieved for the user's question.
6. Groq LLM generates an answer using the retrieved context.

## ⚠️ Limitations

- The application currently processes one uploaded PDF at a time.
- Scanned PDFs without a text layer may not work correctly.
- Answers depend on the quality and content of the uploaded policy.
- This application is a learning project and should not replace official HR guidance.

## 🚀 Future Improvements

- Support multiple PDF documents.
- Add conversation memory.
- Improve document source citations.
- Add HR policy categories.
- Deploy on Streamlit Community Cloud.
- Add authentication and access control.

## 👩‍💻 Author

Hafsa Tofique

## 📄 License

This project is created for educational and portfolio purposes.
