import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Define model configurations
models = {
    "minilm": {
        "model": SentenceTransformer("all-MiniLM-L6-v2"),
        "json_path": "outputs/task2_embedding_minilm_window_75.json",
        "embedding_key": "embedding_minilm",
        "embedding_dim": 384
    },
    "distilbert": {
        "model": SentenceTransformer("distilbert-base-nli-stsb-mean-tokens"),
        "json_path": "outputs/task2_embedding_distilbert_window_75.json",
        "embedding_key": "embedding_distilbert",
        "embedding_dim": 768
    }
}

# Define queries to test
queries = [
    "Why people feel lonely?",
    "How to improve social skills?",
    "What makes someone a good communicator?"
]

# Initialize final results dictionary
final_results = {}

# Process each model
for model_name, config in models.items():
    print(f"Processing model: {model_name}")
    # Load model and data
    embedding_model = config["model"]
    with open(config["json_path"], "r", encoding="utf-8") as f:
        data = json.load(f)

    # Prepare embeddings and metadata
    embedding_matrix = []
    metadata = []
    for item in data:
        embedding_matrix.append(item[config["embedding_key"]])
        metadata.append({
            "chunk_index": item.get("chunk_index", None),
            "text": item["text"]
        })

    embedding_matrix = np.array(embedding_matrix).astype("float32")
    faiss.normalize_L2(embedding_matrix)

    # Create FAISS index
    index = faiss.IndexFlatIP(config["embedding_dim"])
    index.add(embedding_matrix)

    # Store results for this model
    model_results = {}

    # Process each query
    for query in queries:
        query_vec = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)
        top_k = 3
        scores, indices = index.search(query_vec, top_k)

        # Map results
        query_results = []
        for score, idx in zip(scores[0], indices[0]):
            query_results.append({
                "chunk_index": metadata[idx]["chunk_index"],
                "text": metadata[idx]["text"],
                "score": float(score)
            })

        model_results[query] = query_results

    final_results[model_name] = model_results

# Save results to JSON
output_path = "outputs/task2_compare_models.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Results saved to {output_path}")
