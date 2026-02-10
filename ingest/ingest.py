import os
import sys

# This helps Python find the 'core' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# --- CONFIG ---
DATA_DIR = "./data"  # Make sure your PDFs are here
COLLECTION_NAME = "large_dataset_v1"

# --- GLOBAL SETTINGS ---
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=150)

def run_master_ingest():
    print(f"📂 Scanning directory: {DATA_DIR}...")
    
    # 1. Read Files
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Directory '{DATA_DIR}' not found. Please create it and add PDFs.")
        return

    reader = SimpleDirectoryReader(input_dir=DATA_DIR, recursive=True)
    documents = reader.load_data()
    print(f"✅ Found {len(documents)} pages.")

    # 2. Connect to Qdrant (Docker)
    print("🧠 Connecting to Qdrant...")
    client = QdrantClient(host="localhost", port=6333)
    vector_store = QdrantVectorStore(collection_name=COLLECTION_NAME, client=client)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Index Data
    print("🚀 Indexing data (this may take a while)...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    print("✅ SUCCESS! Data indexed.")

if __name__ == "__main__":
    run_master_ingest()