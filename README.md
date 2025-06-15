# Mini RAG System: YouTube Transcript Retrieval-Augmented Generation

## 📘 Background and Motivation

Large Language Models (LLMs) like GPT-4 are powerful tools for question answering and summarization. However, they often hallucinate facts when operating outside their training domain or when handling very specific user queries. To overcome this, **Retrieval-Augmented Generation (RAG)** combines LLMs with an external knowledge source (e.g., YouTube transcripts, documents, databases), improving both factual correctness and domain relevance.

This project aims to simulate a **mini RAG system** that:
- Uses preprocessed **YouTube transcripts** as the knowledge base.
- Performs **semantic retrieval** using FAISS and Sentence Transformers.
- Generates responses using either **a local GPT-2 model** or the **GPT-4 API**.
- Benchmarks and compares performance using different encoder models (MiniLM vs. DistilBERT).

---

## 🛠 Methodologies

### 1. 📂 Data Preparation (Task 1)

- All `.txt` files in the `datasets/` folder are treated as raw YouTube transcript data. 
- Each file undergoes:
  - **Cleaning**: Lowercasing, punctuation removal, and stopword filtering using NLTK.
  - **Tokenization**: Word tokenization followed by windowed chunking.
  - **Chunking**: Fixed-size sliding window with overlap (e.g., 75 tokens with 50-token overlap).
- Output is saved to `outputs/task1_cleaned_window_75.json`.

### 2. 🔍 Embedding and Vector Storage (Task 2)

- Chunks are encoded using two different SentenceTransformer models (comparison saved in `outputs/task2_compare_models.json):
  - `all-MiniLM-L6-v2` (384-dim)
  - `distilbert-base-nli-stsb-mean-tokens` (768-dim)
- Embeddings are stored in a FAISS index for efficient cosine similarity search.
- Metadata (e.g., chunk index and original text) is preserved in a JSON format.

### 3. 🔎 Retrieval (Task 2)

- Given a user query, it's embedded using the same model.
- FAISS retrieves top-k relevant chunks based on inner product similarity (after L2 normalization).
- Results are returned along with their similarity scores and text.

### 4. 💬 Multi-turn Query Handling (Task 3)

- Maintains a **query history buffer** to handle follow-up questions.
- Context-aware queries are embedded and used for retrieval, improving relevance in multi-turn interactions.

### 5. 🧠 Response Generation (Task 3)

- Retrieved chunks form the context for answering the user’s question.
- Two generation options:
  - **GPT-2 (Local)**: Loaded from local disk using Hugging Face Transformers.
  - **GPT-4 (API)**: Uses OpenAI ChatCompletion endpoint.
- Prompt = `"Context: [chunks] \n\nQuestion: [query] \nAnswer:"`

### 6. ⏱ Performance Benchmarking (Task 3)

- Latency for retrieval and generation is measured per query.
- Results include total time and model used.
- Final performance metrics saved to `outputs/task3_rag_performance_gpt2.json` and `outputs/task3_rag_performance_gpt4.json`.

---

## 📊 Features

- ✅ Modular pipeline (cleaning, embedding, retrieval, generation)
- ✅ Model comparison: MiniLM vs. DistilBERT
- ✅ Retrieval + generation fusion (RAG)
- ✅ Local and API-based LLM support
- ✅ Latency and performance evaluation
- ✅ Multi-turn query support (optional context adaptation)

---

## 📌 Requirements

- Python 3.8+
- `sentence-transformers`
- `transformers`
- `faiss-cpu`
- `nltk`
- `torch`
- (Optional) `openai`

---

## 🚀 How to Run

```bash
# Step 1: Install requirements
pip install -r requirements.txt

# Step 2: Clean and chunk transcripts
python task1_ingest_clean_chunk.py

# Step 3: Embed and store vectors
python task2_1_embedding.py

# Step 4: Run RAG pipeline and benchmark
python task3_rag_and_scaling.py


