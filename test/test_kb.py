import pytest
from fastapi.testclient import TestClient
from app.main import app, STORE

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}

def test_ingest_and_search_and_qa():
    STORE.clear()
    text = "OpenAI was founded in 2015. The founders include Sam Altman and others."
    r = client.post("/ingest", json={"id": "doc_test", "text": text})
    assert r.status_code == 201
    data = r.json()
    assert data["id"] == "doc_test"
    assert data["chunks"] >= 1

    r2 = client.get("/search", params={"q": "founders of OpenAI", "top_k": 3})
    assert r2.status_code == 200
    sdata = r2.json()
    assert sdata["query"] == "founders of OpenAI"
    assert len(sdata["results"]) > 0

    r3 = client.post("/qa", json={"question": "Who founded OpenAI?", "top_k": 3})
    assert r3.status_code == 200
    qdata = r3.json()
    assert "OpenAI" in qdata["answer"] or len(qdata["sources"]) > 0

def test_completeness():
    STORE.clear()
    text = "This document talks about Kubernetes and containers."
    client.post("/ingest", json={"id": "doc2", "text": text})
    r = client.post("/completeness", json={"question": "What is Kubernetes?", "top_k": 2, "threshold": 0.01})
    assert r.status_code == 200
    data = r.json()
    assert "complete" in data
