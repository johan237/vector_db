import numpy as np
from typing import Dict, Any, List, Tuple, Callable, Optional


class SchemaVectorDB:
    """
    Schema-driven Vector DB.
    Supports multiple embedding groups (title+desc, content chunks, images, etc.).
    Stores chunk metadata for better document-level search.
    """

    def __init__(self, schema: Dict[str, Any]):
        """
        schema format:
        {
            "embed_fields": [
                {
                    "fields": ["title", "description"],  # combined into one text
                    "embedding_model": model_instance,         # user passes actual model (e.g., SentenceTransformer)
                    "dim": 384,
                    "chunked": False
                },
                {
                    "fields": ["content"],
                    "embedding_model": model_instance,         # bigger model for long text
                    "dim": 1024,
                    "chunked": True,
                    "chunk_size": 300,
                    "overlap": 50
                }
            ],
            "filters": ["author", "tags"]
        }
        """
        self.schema = schema
        # store encoder models as passed by user
        self.encoders = {i: group["embedding_model"] for i, group in enumerate(schema.get("embed_fields", []))}
        # main index storage
        self.indexes: Dict[int, Dict[str, Any]] = {}

    # ---------- Internal helpers ----------
    def _chunk_text(self, text: str, size: int, overlap: int) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata (offsets)."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append({
                "text": text[start:end],
                "start": start,
                "end": end,
                "page": None  # can be filled if page info is available from parser
            })
            start += size - overlap
        return chunks

    def _build_group_chunks(self, info: Dict[str, Any], group: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare chunk dicts for one embed group."""
        fields = group["fields"]
        concat_text = " ".join([str(info.get(f, "")) for f in fields if f in info])

        if group.get("chunked", False):
            return self._chunk_text(
                concat_text,
                group.get("chunk_size", 200),
                group.get("overlap", 50)
            )
        else:
            return [{"text": concat_text, "start": 0, "end": len(concat_text), "page": None}]

    # ---------- Core methods ----------
    def add(self, uid: int, info: Dict[str, Any]):
        """Add item: generate embeddings for all groups, store chunk metadata."""
        group_vectors = {}
        group_chunks = {}

        for i, group in enumerate(self.schema["embed_fields"]):
            chunks = self._build_group_chunks(info, group)
            texts = [c["text"] for c in chunks]
            encoder = self.encoders[i]
            vecs = encoder.encode(texts).astype(np.float32)

            group_vectors[i] = vecs
            group_chunks[i] = chunks

        self.indexes[uid] = {
            "groups": group_vectors,   # vectors
            "chunks": group_chunks,    # chunk metadata
            "info": info               # original info dict
        }

    def add_many(self, items: Dict[int, Dict[str, Any]]):
        """Batch add items for speed."""
        for uid, info in items.items():
            self.add(uid, info)

    def update(self, uid: int, new_info: Dict[str, Any]) -> bool:
        """Update entry with new info and regenerate embeddings."""
        if uid not in self.indexes:
            return False
        self.add(uid, new_info)
        return True

    def delete(self, uid: int) -> bool:
        """Delete entry by uid."""
        if uid in self.indexes:
            del self.indexes[uid]
            return True
        return False

    # ---------- Search ----------
    def search_in_document(
        self,
        uid: int,
        query: str,
        field_group: int,
        top_k: int = 3,
        method: str = "cosine"
    ) -> List[Dict[str, Any]]:
        """
        Search inside a single document (chunk-level).
        Returns best matching chunks with metadata.
        """
        if uid not in self.indexes:
            return []

        query_vec = self.encoders[field_group].encode([query])[0].astype(np.float32)

        # available scoring functions
        def cosine(a, b):
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            return float(np.dot(a, b) / denom) if denom > 0 else 0.0

        def dot(a, b):
            return float(np.dot(a, b))

        def euclidean(a, b):
            return -float(np.linalg.norm(a - b))

        score_fn = {"cosine": cosine, "dot": dot, "euclidean": euclidean}[method]

        entry = self.indexes[uid]
        vecs = entry["groups"][field_group]
        chunks = entry["chunks"][field_group]

        sims = []
        for i, vec in enumerate(vecs):
            score = score_fn(query_vec, vec)
            meta = chunks[i].copy()
            meta["score"] = score
            sims.append(meta)

        sims.sort(key=lambda x: x["score"], reverse=True)
        return sims[:top_k]
