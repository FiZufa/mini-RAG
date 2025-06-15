import faiss
import numpy as np

def build_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)  # L2 ~ cosine if vectors normalized
    index.add(np.array(embeddings).astype("float32"))
    return index

def retrieve_chunks(query, model_name="minilm", top_k=3):
    # Load data and model
    df = pd.read_json(f"embeddings_{model_name}.json")
    model = SentenceTransformer({
        "minilm": "all-MiniLM-L6-v2",
        "distilbert": "distilbert-base-nli-stsb-mean-tokens"
    }[model_name])

    query_vec = model.encode([query])
    embeddings = np.vstack(df[f"embedding_{model_name}"])

    # Build and search index
    index = build_faiss_index(embeddings)
    D, I = index.search(np.array(query_vec).astype("float32"), top_k)

    results = []
    for i in I[0]:
        result = {
            "document": df.iloc[i]["document"],
            "chunk_index": df.iloc[i]["chunk_index"],
            "text": df.iloc[i]["text"]
        }
        results.append(result)

    return results
