import os
import shutil
import fitz
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from qdrant_client import QdrantClient


DATA_DIR = "data"
IMAGE_OUTPUT_DIR = "frontend/static/extracted_images"
COLLECTION_NAME = "large_dataset_v1"

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434"
)


def clean_filename(text):
    clean = "".join([c if c.isalnum() else "_" for c in text])
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_")


def find_caption_below_image(page, img_rect):
    blocks = page.get_text("blocks")
    img_x0, img_y0, img_x1, img_y1 = img_rect

    closest_caption = None
    min_distance = 300

    for b in blocks:
        b_x0, b_y0, b_x1, b_y1, text, _, _ = b
        text = text.strip()

        if "figure" in text.lower():
            vertical_distance = abs(b_y0 - img_y1)
            if vertical_distance < min_distance:
                min_distance = vertical_distance
                closest_caption = text

    if closest_caption:
        return closest_caption.replace("\n", " ")

    return None


def extract_content():
    documents = []

    if os.path.exists(IMAGE_OUTPUT_DIR):
        shutil.rmtree(IMAGE_OUTPUT_DIR)
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(DATA_DIR, filename)
            doc = fitz.open(pdf_path)

            for page_num, page in enumerate(doc):
                text = page.get_text()
                image_list = page.get_images(full=True)
                image_tags = ""

                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        img_rects = page.get_image_rects(xref)
                        if not img_rects:
                            continue

                        img_rect = img_rects[0]
                        caption = find_caption_below_image(page, img_rect)

                        if caption:
                            safe_name = clean_filename(caption)
                        else:
                            safe_name = f"{clean_filename(filename[:-4])}_p{page_num+1}_i{img_index+1}"

                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        image_name = f"{safe_name}.{image_ext}"
                        image_path = os.path.join(IMAGE_OUTPUT_DIR, image_name)

                        with open(image_path, "wb") as f:
                            f.write(image_bytes)

                        web_path = f"/static/extracted_images/{image_name}"

                        image_tags += (
                            f"\n[IMAGE_REF: {web_path}] "
                            f"(Contains: {safe_name.replace('_', ' ')} Diagram Figure)\n"
                        )

                    except Exception as e:
                        print("Image error:", e)

                final_page_text = image_tags + "\n" + text

                doc_obj = Document(
                    text=final_page_text,
                    metadata={"file_name": filename, "page_number": page_num + 1}
                )

                documents.append(doc_obj)

    return documents


def main():
    client = QdrantClient(host="localhost", port=6333, timeout=300)

    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)

    documents = extract_content()

    vector_store = QdrantVectorStore(
        collection_name=COLLECTION_NAME,
        client=client,
        batch_size=32
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    print("Database updated successfully.")


if __name__ == "__main__":
    main()
