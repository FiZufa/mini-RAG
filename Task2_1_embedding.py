from sentence_transformers import SentenceTransformer
import json
import pandas as pd
import os

# Load cleaned chunks
with open("outputs/task1_cleaned_window_75.json", "r", encoding="utf-8") as f:
    cleaned_data = json.load(f)

# Choose two models
models = {
    "distilbert": SentenceTransformer("distilbert-base-nli-stsb-mean-tokens"),
    "minilm": SentenceTransformer("all-MiniLM-L6-v2")
}

# Flatten chunks and track metadata
data = []
for doc_name, chunks in cleaned_data.items():
    for idx, chunk in enumerate(chunks):
        data.append({
            "document": doc_name,
            "chunk_index": idx,
            "text": chunk
        })

df = pd.DataFrame(data)

# Generate embeddings for each model and save
for model_name, model in models.items():
    print(f"🔄 Generating embeddings with: {model_name}")
    df[f"embedding_{model_name}"] = model.encode(df["text"].tolist(), show_progress_bar=True).tolist()
    df.to_json(f"outputs/task2_embedding_{model_name}_window_75.json", orient="records", indent=2)

print("✅ Embeddings saved!")
