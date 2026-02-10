import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from qdrant_client import QdrantClient
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core.memory import ChatMemoryBuffer

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

Settings.llm = Ollama(
    model="llama3.2:1b", 
    request_timeout=120.0, 
    context_window=4096, 
    temperature=0.1 
)

def get_chat_engine():
    client = QdrantClient(host="localhost", port=6333)
    vector_store = QdrantVectorStore(collection_name="large_dataset_v1", client=client)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    rerank_postprocessor = SentenceTransformerRerank(
        model="BAAI/bge-reranker-base", 
        top_n=5
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    system_prompt = (
        "You are a helpful assistant for the user's documents. "
        "Use the context provided below to answer the user's question.\n"
        "GUIDELINES:\n"
        "1. If the context contains the answer, explain it clearly.\n"
        "2. If the context mentions the topic but is incomplete, summarize what is there.\n"
        "3. Only say 'I don't know' if the context is completely unrelated.\n"
        "4. Be polite and professional."
    )

    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=system_prompt,
        similarity_top_k=15, 
        node_postprocessors=[rerank_postprocessor],
        streaming=True 
    )
    
    return chat_engine