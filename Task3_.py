import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# === Load Embedding Model ===
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# === Load Local GPT-2 Model & Tokenizer ===
gpt2_path = r"D:\IMPORTANT\final_semester\modelscope_cache\models\openai-community\gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path, local_files_only=True)
model = GPT2LMHeadModel.from_pretrained(gpt2_path, local_files_only=True)
model.eval()

# === Load Embedding Data (MiniLM) ===
with open("outputs/task2_embedding_minilm_window_75.json", "r", encoding="utf-8") as f:
    data = json.load(f)

embedding_dim = 384
embedding_matrix = []
metadata = []

for item in data:
    embedding_matrix.append(item["embedding_minilm"])
    metadata.append({
        "chunk_index": item.get("chunk_index", None),
        "text": item["text"]
    })

embedding_matrix = np.array(embedding_matrix).astype("float32")
faiss.normalize_L2(embedding_matrix)
index = faiss.IndexFlatIP(embedding_dim)
index.add(embedding_matrix)

# === Retrieval Function with Context ===
def retrieve_chunks(query, history=None, top_k=3):
    if history:
        query = " ".join(history + [query])
    query_vec = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "chunk_index": metadata[idx]["chunk_index"],
            "text": metadata[idx]["text"],
            "score": float(score)
        })
    return results

# === RAG: Generate Answer from Retrieved Chunks ===
def generate_answer(query, retrieved_chunks):
    context = "\n".join([chunk["text"] for chunk in retrieved_chunks])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=200, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.replace(prompt, "").strip()

# === Main: Multi-Turn Example ===
if __name__ == "__main__":
    query_history = []
    queries = [
        "Why do people feel lonely?",
        "What skills can help with that?",
        "How can someone practice those skills?"
    ]

    for turn, q in enumerate(queries, 1):
        retrieved = retrieve_chunks(q, history=query_history)
        answer = generate_answer(q, retrieved)
        print(f"Turn {turn} — Query: {q}")
        print(f"Answer: {answer}\n{'='*60}\n")
        query_history.append(q)
