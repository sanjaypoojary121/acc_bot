# 🧠 Private RAG Chatbot (Local AI)

> **A secure, 100% local document Chatbot built with Llama 3.2, Qdrant, and Flask.**

This project allows you to chat with your own PDF documents privately. It runs entirely on your local machine—no data is sent to the cloud. It uses **Retrieval-Augmented Generation (RAG)** to find relevant answers from your files and generate accurate responses using a local LLM.

---

## ✨ Features

* **🔒 100% Private:** Runs locally on your hardware. No API keys or cloud costs.
* **🧠 Local LLM Power:** Uses **Llama 3.2 (1B)** via Ollama for fast, intelligent answers.
* **⚡ Fast Retrieval:** Uses **Qdrant** (Vector Database) & **Re-Ranking** for high-accuracy search.
* **💬 Streaming Interface:** Real-time typing effect like ChatGPT.
* **📂 Easy Data Ingestion:** Automatically scans and indexes all PDFs in your data folder.
* **📚 Citation Support:** Shows exactly which document and page the answer came from.

---

## 🛠️ Tech Stack

* **Frontend:** HTML5, CSS3 (Modern Chat UI), JavaScript (Fetch API & Streaming)
* **Backend:** Python 3.11+, Flask
* **AI Engine:** LlamaIndex (Framework), Ollama (LLM Runner)
* **Database:** Qdrant (Vector Store via Docker)
* **Models:**
    * **LLM:** `llama3.2:1b` (Optimized for speed/memory)
    * **Embeddings:** `nomic-embed-text`
    * **Re-Ranker:** `BAAI/bge-reranker-base`

---

## 🚀 Installation Guide

### Prerequisites
1.  **Python 3.11** or higher.
2.  **Docker Desktop** (For the Qdrant database).
3.  **Ollama** (Download from [ollama.com](https://ollama.com)).

### Step 1: Clone & Setup Environment
```bash
# Clone the repository
git clone [https://github.com/yourusername/local-rag-chatbot.git](https://github.com/yourusername/local-rag-chatbot.git)
cd local-rag-chatbot

# Create a virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate
# Activate it (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

Step 2: Prepare AI Models
Open your terminal and pull the required models:

Bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
Step 3: Start the Database
Run Qdrant using Docker:

Bash
docker run -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
(Keep this terminal window open!)

🏃‍♂️ Usage
1. Load Your Documents
Place your PDF files into the data/ folder. Then run the ingestion script:

Bash
python ingest/ingest.py
Wait for the "✅ SUCCESS! Data indexed" message.

2. Start the Chat Server
Bash
python app.py
3. Chat!
Open your browser and go to: 👉 http://localhost:5000

📂 Project Structure
Plaintext
📁 local-rag-chatbot/
├── 📁 core/
│   └── engine.py         # The AI Brain (LlamaIndex logic)
├── 📁 data/              # Place your PDF files here
├── 📁 frontend/
│   ├── 📁 static/        # CSS styles
│   └── 📁 templates/     # HTML interface
├── 📁 ingest/
│   └── ingest.py         # Script to load PDFs into database
├── .gitignore
├── app.py                # Flask Web Server
└── requirements.txt      # Python dependencies
🔧 Troubleshooting
"Model requires more memory": Make sure you pulled llama3.2:1b (the small version) and updated core/engine.py to use it.

"Connection Refused": Ensure Docker is running and you see the Qdrant container active.

"Template Not Found": Ensure your app.py is pointing to frontend/templates.