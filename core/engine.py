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
    temperature=0.0,
)


def get_chat_engine():
    client = QdrantClient(host="localhost", port=6333, timeout=60)

    vector_store = QdrantVectorStore(
        collection_name="large_dataset_v1",
        client=client,
    )

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    rerank_postprocessor = SentenceTransformerRerank(
        model="BAAI/bge-reranker-base",
        top_n=2,  
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=2000)

    system_prompt = (
        "You are a STRICT document-only assistant.\n\n"

        "MANDATORY RULES:\n"
        "1. Answer ONLY using retrieved context.\n"
        "2. If answer not found, reply EXACTLY:\n"
        "'No data available in the provided documents.'\n"
        "3. If user asks for a diagram and context contains [IMAGE_REF:], "
        "reply ONLY: 'Please refer to the figure below.'\n"
        "4. Never mention copyright.\n"
        "5. Never say you cannot create images.\n"
        "6. Never use outside knowledge.\n"
        "7. Maximum 4 lines.\n"
        "8. Never draw ASCII diagrams.\n"
    )

    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=system_prompt,
        similarity_top_k=10,
        node_postprocessors=[rerank_postprocessor],
        response_mode="compact",
        streaming=True,
    )

    return chat_engine
