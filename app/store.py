import os, json, math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import embeddings as emb_mod

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

class SimpleVectorStore:
    def __init__(self, persist: bool = True):
        self.docs: Dict[str, Dict[str, Any]] = {}  # doc_id -> {raw_text, chunks: [ids]}
        self.chunks: List[Dict[str, Any]] = []  # each: {id, doc_id, text}
        self._use_openai = emb_mod.USE_OPENAI
        self._embeddings = None  # numpy array (n, d) when openai in use
        self._tfidf = TfidfVectorizer(stop_words='english')
        self._tfidf_matrix = None
        self.persist = persist
        os.makedirs(DATA_DIR, exist_ok=True)
        if persist:
            self._load()

    def _persist(self):
        docs_path = os.path.join(DATA_DIR, 'docs.json')
        chunks_path = os.path.join(DATA_DIR, 'chunks.json')
        emb_path = os.path.join(DATA_DIR, 'embeddings.npy')
        try:
            with open(docs_path, 'w', encoding='utf-8') as f:
                json.dump(self.docs, f, ensure_ascii=False, indent=2)
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
            if self._embeddings is not None:
                np.save(emb_path, self._embeddings)
            print(f"✅ Persisted docs to {DATA_DIR}")
        except Exception as e:
            raise RuntimeError(f"Persist failed: {e}")


    def _load(self):
        try:
            docs_path = os.path.join(DATA_DIR, 'docs.json')
            chunks_path = os.path.join(DATA_DIR, 'chunks.json')
            emb_path = os.path.join(DATA_DIR, 'embeddings.npy')
            if os.path.exists(docs_path):
                with open(docs_path, 'r', encoding='utf-8') as f:
                    self.docs = json.load(f)
            if os.path.exists(chunks_path):
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    self.chunks = json.load(f)
            if os.path.exists(emb_path):
                self._embeddings = np.load(emb_path)
            else:
                self._rebuild_index()
        except Exception as e:
            print('Load error', e)

    def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        # Naive chunking: split by paragraphs then by fixed char windows
        parts = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        for p in parts:
            if len(p) <= chunk_size:
                chunks.append(p)
            else:
                for i in range(0, len(p), chunk_size):
                    chunks.append(p[i:i+chunk_size])
        if not chunks:
            # fallback split by char windows
            for i in range(0, len(text), chunk_size):
                chunks.append(text[i:i+chunk_size])
        return chunks

    def add_document(self, doc_id: str, text: str) -> int:
        """Ingests a document, stores raw and chunks, rebuilds embeddings/index incrementally."""
        if doc_id in self.docs:
            # support incremental updates by appending new text
            self.docs[doc_id]['raw'] += '\n\n' + text
        else:
            self.docs[doc_id] = {'raw': text, 'chunks': []}
        chunks = self._chunk_text(text)
        start_idx = len(self.chunks)
        for i, c in enumerate(chunks):
            cid = f"{doc_id}__{len(self.chunks)}"
            self.chunks.append({'id': cid, 'doc_id': doc_id, 'text': c})
            self.docs[doc_id]['chunks'].append(cid)
        # rebuild embeddings/index (simple approach)
        self._rebuild_index()
        if self.persist:
            self._persist()
        return len(chunks)

    def _rebuild_index(self):
        texts = [c['text'] for c in self.chunks]
        if not texts:
            self._embeddings = None
            self._tfidf_matrix = None
            return
        if self._use_openai:
            try:
                embs = emb_mod.embed_texts(texts)
                self._embeddings = np.array(embs, dtype=float)
            except Exception as e:
                print('OpenAI embed failed, falling back to TF-IDF', e)
                self._use_openai = False
        if not self._use_openai:
            self._tfidf_matrix = self._tfidf.fit_transform(texts)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        texts = [c['text'] for c in self.chunks]
        if self._use_openai and self._embeddings is not None:
            q_emb = np.array(emb_mod.embed_texts([query]), dtype=float)
            sims = cosine_similarity(q_emb, self._embeddings)[0]
            idxs = list(reversed(sims.argsort()))[:top_k]
            return [{'id': self.chunks[i]['id'], 'doc_id': self.chunks[i]['doc_id'], 'text': self.chunks[i]['text'], 'score': float(sims[i])} for i in idxs]
        else:
            if not hasattr(self, '_tfidf_matrix') or self._tfidf_matrix is None:
                return []
            qv = self._tfidf.transform([query])
            sims = cosine_similarity(qv, self._tfidf_matrix)[0]
            idxs = list(reversed(sims.argsort()))[:top_k]
            return [{'id': self.chunks[i]['id'], 'doc_id': self.chunks[i]['doc_id'], 'text': self.chunks[i]['text'], 'score': float(sims[i])} for i in idxs]

    def clear(self):
        self.docs = {}
        self.chunks = []
        self._embeddings = None
        self._tfidf_matrix = None
        if self.persist:
            self._persist()

# single global store
STORE = SimpleVectorStore(persist=True)
