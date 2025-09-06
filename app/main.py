from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
import os

from .models import IngestRequest, IngestResponse, SearchResponse, SearchItem, QARequest, QAResponse, CompletenessRequest, CompletenessResponse
from .store import STORE

app = FastAPI(title="WandAI KB Prototype", version="0.1.0")

@app.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest(req: IngestRequest):
    doc_id = req.id or f"doc_{len(STORE.docs)+1}"
    chunks = STORE.add_document(doc_id, req.text)

    # Force persist explicitly
    STORE._persist()

    return IngestResponse(id=doc_id, chunks=chunks)


@app.get("/search", response_model=SearchResponse)
async def search(q: str, top_k: int = 5):
    results = STORE.search(q, top_k=top_k)
    items = [SearchItem(id=r['id'], doc_id=r['doc_id'], score=r['score'], text=r['text']) for r in results]
    return SearchResponse(query=q, results=items)

@app.post("/qa", response_model=QAResponse)
async def qa(req: QARequest):
    results = STORE.search(req.question, top_k=req.top_k)
    if not results:
        raise HTTPException(status_code=404, detail="no documents indexed")
    # naive answer: concatenate top contexts (<= 2000 chars)
    combined = "\n\n".join([r['text'] for r in results])
    answer = combined[:2000] + ("..." if len(combined) > 2000 else "")
    sources = [SearchItem(id=r['id'], doc_id=r['doc_id'], score=r['score'], text=r['text']) for r in results]
    return QAResponse(question=req.question, answer=answer, sources=sources)

@app.post("/completeness", response_model=CompletenessResponse)
async def completeness(req: CompletenessRequest):
    results = STORE.search(req.question, top_k=req.top_k)
    if not results:
        return CompletenessResponse(question=req.question, complete=False, avg_score=0.0, top_k=req.top_k)
    avg = sum(r['score'] for r in results) / len(results)
    complete = avg >= req.threshold
    return CompletenessResponse(question=req.question, complete=complete, avg_score=avg, top_k=req.top_k)

@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/")
async def root():
    return {"message": "Welcome to WandAI Challenge 2 KB API — see /docs for interactive docs"}
