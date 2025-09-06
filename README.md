# Wand AI Backend Engineer – Challenge 2: Knowledge Base Prototype

A minimal prototype of a **knowledge base backend** built with  
**Python + FastAPI + scikit-learn TF-IDF (with optional OpenAI embeddings)**.  

This service ingests documents, indexes them into a vector store, and supports:
- Document ingestion & persistence
- Semantic search
- Question answering (QA) over ingested content
- Completeness scoring for coverage assessment

---

## Features

- **Ingest documents** → Texts are chunked and stored with embeddings (OpenAI or TF-IDF).
- **Search** → Retrieve top-k similar chunks for a query.
- **QA** → Naive answer synthesis from top results.
- **Completeness** → Checks whether a query is sufficiently covered in the KB.
- **Persistence** → Docs, chunks, and embeddings are persisted to `data/`.
- **Extensible** → Plug in new embedding models or chunking strategies easily.

---

## API Endpoints

| Method | Endpoint        | Description |
|--------|-----------------|-------------|
| POST   | `/ingest`       | Ingest a new document into the KB |
| GET    | `/search`       | Search documents by query |
| POST   | `/qa`           | Ask a question, get synthesized answer + sources |
| POST   | `/completeness` | Check whether a question is "complete" given threshold |
| GET    | `/health`       | Health check |

---

## Running Locally

**Requirements:** Python 3.11+

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# Open http://localhost:8000/docs
```
## Running with Docker
```bash
docker build -t wandai-kb .
docker compose up --build
# Service runs on http://localhost:8001
```
## Example Usage
**Ingest a Document**
```bash
curl -X POST http://localhost:8001/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc1",
    "text": "OpenAI was founded in 2015. Sam Altman is one of the founders."
  }'
```
## Search
```bash
curl -X GET "http://localhost:8001/search?q=founders of OpenAI&top_k=3"
```
## QA
```
curl -X POST http://localhost:8001/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "Who founded OpenAI?", "top_k": 3}'
```

## Testing

**Run unit tests with pytest:**
```bash
pytest -v
```
* Tests included in test/test_kb.py cover:
* Health check
* Ingest & persistence
* Search results relevance
* QA response with sources
* Completeness scoring

## Approach

**Document Store → SimpleVectorStore manages docs, chunks, and embeddings.**

## Embeddings:

* Uses OpenAI embeddings if OPENAI_API_KEY is set.
* Falls back to TF-IDF vectorization (scikit-learn) otherwise.
* Chunking → Splits text by paragraphs, then by fixed windows if needed.
* Search → Cosine similarity between query and indexed embeddings.
* QA → Combines top-k retrieved chunks (up to 2000 chars).
* Completeness → Average similarity ≥ threshold → considered “complete”.

## Example Test Workflow
```bash
def test_ingest_and_search_and_qa():
    STORE.clear()
    text = "OpenAI was founded in 2015. The founders include Sam Altman."
    client.post("/ingest", json={"id": "doc_test", "text": text})
    r2 = client.get("/search", params={"q": "founders of OpenAI", "top_k": 3})
    assert len(r2.json()["results"]) > 0
    r3 = client.post("/qa", json={"question": "Who founded OpenAI?", "top_k": 3})
    assert "OpenAI" in r3.json()["answer"]
```
## Design Decisions & Trade-offs

* Fallback embeddings: OpenAI for high-quality vectors, TF-IDF for local/dev mode.
* Naive QA: concatenation of top chunks (kept simple for prototype).
* Completeness metric: average similarity vs threshold; can be improved with better scoring.
* Persistence: JSON + .npy for simplicity; swap with DB (Postgres/Redis) in prod.
* Scalability: Single-process, suitable for prototype; production can use FAISS + workers.

## Demo Script (≤5 min)

* Start the API with Docker (docker compose up).
* Ingest the sample book (populate_vd.py).
* Run search queries → see relevant results.
* Run QA → answer is synthesized from top docs.
* Run completeness → check coverage.
* Show persistence in data/ folder. 
* Completeness
```bash
curl -X POST http://localhost:8001/completeness \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Kubernetes?", "top_k": 2, "threshold": 0.2}'
```