import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from functools import lru_cache
import torch
from openai import OpenAI

# === Configuration ===
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_JSON_PATH = "outputs/task2_embedding_minilm_window_75.json"

USE_OPENAI_API = True  
GPT2_MODEL_PATH = "D:/IMPORTANT/final_semester/modelscope_cache/models/openai-community/gpt2"

# === Load models ===
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

if USE_OPENAI_API:
    client = OpenAI(api_key="***")

else:
    tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(GPT2_MODEL_PATH)
    model.eval()

# === Load Embeddings and Metadata ===
with open(EMBEDDING_JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

embedding_matrix = []
metadata = []

for item in data:
    embedding_matrix.append(item["embedding_minilm"])
    metadata.append({
        "chunk_index": item.get("chunk_index"),
        "text": item["text"]
    })

embedding_matrix = np.array(embedding_matrix).astype("float32")
faiss.normalize_L2(embedding_matrix)

# === Build FAISS index ===
index = faiss.IndexFlatIP(EMBEDDING_DIM)
index.add(embedding_matrix)

# === Helper Functions ===
@lru_cache(maxsize=128)
def embed_query(query):
    vec = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vec)
    return vec

def retrieve_chunks(query, top_k=3):
    query_vec = embed_query(query)
    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "chunk_index": metadata[idx]["chunk_index"],
            "text": metadata[idx]["text"],
            "score": float(score)
        })
    return results

def generate_answer(query, retrieved_chunks):
    context = "\n".join(chunk["text"] for chunk in retrieved_chunks)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

    if USE_OPENAI_API:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content

    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id, 
                attention_mask=torch.ones_like(input_ids)  
            )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text[len(prompt):].strip()


def benchmark_rag_pipeline(queries):
    results = []
    for query in queries:
        print(f"Processing query: {query}\n")

        start_retrieve = time.time()
        chunks = retrieve_chunks(query)
        retrieve_time = time.time() - start_retrieve

        start_gen = time.time()
        answer = generate_answer(query, chunks)
        gen_time = time.time() - start_gen

        results.append({
            "query": query,
            "retrieval_time_sec": round(retrieve_time, 4),
            "generation_time_sec": round(gen_time, 4),
            "total_time_sec": round(retrieve_time + gen_time, 4),
            "retrieved_chunks": chunks,
            "generated_answer": answer
        })
    return results

# === Run benchmark and save results ===
if __name__ == "__main__":
    sample_queries = [
        "Why do people feel lonely?",
        "How can someone improve their social skills?",
        "What is the most important skill for personal growth?"
    ]

    output = benchmark_rag_pipeline(sample_queries)

    with open("outputs/task5_6_rag_performance_gpt4.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("Benchmark results saved to task5_6_rag_performance_gpt4.json")
