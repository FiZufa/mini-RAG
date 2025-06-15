import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load the same embedding model used previously
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 1: Load JSON and extract embeddings + metadata
with open("outputs/task2_embeddings_minilm.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract embeddings and metadata
embedding_dim = 384  # MiniLM
embedding_matrix = []
metadata = []

for item in data:
    embedding_matrix.append(item["task2_embedding_minilm"])
    metadata.append({
        "chunk_index": item.get("chunk_index", None),
        "text": item["text"]
    })

embedding_matrix = np.array(embedding_matrix).astype("float32")

# Step 2: Normalize embeddings
faiss.normalize_L2(embedding_matrix)

# Step 3: Create FAISS index
index = faiss.IndexFlatIP(embedding_dim)  # IP = Inner Product (for cosine sim with norm)
index.add(embedding_matrix)

# Step 4: Query Function
def retrieve_chunks(query, top_k=3):
    # Embed and normalize query
    query_vec = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    # Perform similarity search
    scores, indices = index.search(query_vec, top_k)

    # Step 5: Map results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "chunk_id": metadata[idx]["chunk_id"],
            "text": metadata[idx]["text"],
            "score": float(score)
        })

    return results

# Example usage
if __name__ == "__main__":
    query = "What are the most important in social skills?"
    results = retrieve_chunks(query)
    for res in results:
        print(f"Chunk ID: {res['chunk_id']}, Score: {res['score']:.4f}")
        print(f"Text: {res['text']}\n")