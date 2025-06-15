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

- All `.txt` files in the `datasets/` folder are treated as raw YouTube transcript data from this (YouTube channel)[https://www.youtube.com/@RobertGreeneOfficial/videos] and generated using this (YouTube video online generator)[https://www.youtranscripts.com/]. 
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

## 📊 Benchmark Results Summary

### 💬 Example: Multi-turn QA with GPT-4 (RAG)

This example demonstrates how our RAG system uses retrieved context from YouTube transcripts and combines it with GPT-4 to answer a sequence of related questions.

#### 1. **Q: Why do people feel lonely?**

- ⏱ Retrieval Time: `0.0807 sec`, Generation Time: `2.8176 sec`
- 🧠 **Answer**:
  > People feel lonely when they lack meaningful social interactions or connections. It may also occur when they feel unwanted, unloved, or not respected by others. This can be due to various reasons such as not being able to connect with people on a deeper level, or not being able to find people whose ideas or beliefs align with theirs.

- 📚 **Top Retrieved Chunk Snippet**:
  > “...alone sometimes actually finding oddly enough kind pleasurable... loneliness loses lot staining thing force develop social muscle...”

---

#### 2. **Q: How can someone improve their social skills?**

- ⏱ Retrieval Time: `0.0639 sec`, Generation Time: `3.7998 sec`
- 🧠 **Answer**:
  > Someone can improve their social skills by first recognizing the importance of these skills in daily life. They should practice more often by interacting with various people in different settings... understanding how to work well with others are crucial aspects of improving social skills.

- 📚 **Top Retrieved Chunk Snippet**:
  > “...95% life involves interactions with people... think of social skills like playing piano or shooting free throws—something you can practice and improve.”

---

#### 3. **Q: What is the most important skill for personal growth?**

- ⏱ Retrieval Time: `0.1548 sec`, Generation Time: `1.1904 sec`
- 🧠 **Answer**:
  > The most important skill for personal growth is social skills, such as getting along with people, cooperating, and knowing how to work with others.

- 📚 **Top Retrieved Chunk Snippet**:
  > “...social skills—getting along with people, cooperating, knowing how to work with people—isn’t just innate. It’s something you can and should develop.”

---

## 📝 Conclusion: Insights, Challenges, and Recommendation

### Insights
- Multi-turn question answering using GPT-4 with retrieved context shows coherence and deeper understanding across questions compared with GPT-2.
- Retrieval latency remained low (mostly under 0.2s) using FAISS with sentence embeddings (all-MiniLM-L6-v2).
- Using overlapping token windows (e.g., 75 tokens with 50 overlap) preserved semantic continuity across chunks and improved retrieval relevance.
  
### Challenges
- The dataset used is YouTube video transcript generated by an online generator, which does not contain stopwords, punctuations, and special characters. It is quite challenging to chunk the text without knowing the expressiveness of the text.
- There are many kind of pretrained embeddings, it is challenging to select which pretrained model is suitable for the given task and datasets.
- It is also challenging to measure the quality of the generated response without human involvement.

### Recommendation
- Switch to faster embedding models (e.g., intfloat/e5-small-v2) if scaling to millions of chunks.
- Use local vector store persistence (FAISS + disk I/O) for production environments to avoid recomputing embeddings.
- Replace GPT-2 with a stronger local model (e.g., mistral or gemma) for offline, higher-quality generation.
- Implement contextual memory for handling multi-turn conversations more robustly in the future.
- Improve banchmarking method to evaluate the quality of the response.

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


