import numpy as np
from typing import Dict, Any, List, Tuple, Callable, Union
from vector_db.embedding import EmbeddingFactory


class GenericVectorDB:
    """
    Schema-driven Vector DB.
    Stores {uid: {"embedding": vec, "info": dict}}.
    """

    def __init__(self, schema: Dict[str, List[str]], embedding: str = "sentence-transformers"):
        """
        schema:
          {
            "embed_fields": ["title", "content"],  # fields used for encoding
            "filters": ["author", "tags"]          # fields allowed for filtering
          }
        """
        self.encoder = EmbeddingFactory.get_encoder(embedding)
        self.schema = schema
        self.index: Dict[int, Dict[str, Any]] = {}

    # ---------- Core methods ----------
    def _build_text(self, info: Dict[str, Any]) -> str:
        """Concatenate embed_fields into one string."""
        fields = self.schema.get("embed_fields", [])
        return " ".join([str(info.get(f, "")) for f in fields if f in info])

    def add(self, uid: int, info: Dict[str, Any]):
        """Add a new item to the vector store."""
        text = self._build_text(info)
        vec = self.encoder.encode([text])[0].astype(np.float32)
        self.index[uid] = {"embedding": vec, "info": info}

    def add_many(self, items: Dict[int, Dict[str, Any]]):
        """Batch add items."""
        texts, uids = [], []
        for uid, info in items.items():
            texts.append(self._build_text(info))
            uids.append(uid)

        vecs = self.encoder.encode(texts).astype(np.float32)
        for uid, vec, info in zip(uids, vecs, items.values()):
            self.index[uid] = {"embedding": vec, "info": info}

    def update(self, uid: int, new_info: Dict[str, Any]) -> bool:
        if uid not in self.index:
            return False
        self.add(uid, new_info)
        return True

    def delete(self, uid: int) -> bool:
        if uid in self.index:
            del self.index[uid]
            return True
        return False

    # ---------- Search ----------
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_fn: Callable[[Dict[str, Any]], bool] = None,
        method: str = "cosine",
    ) -> List[Tuple[int, float]]:
        """Search items by similarity with optional filtering."""
        if not self.index:
            return []

        query_vec = self.encoder.encode([query])[0].astype(np.float32)

        # scoring functions
        def cosine(a, b):
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            return float(np.dot(a, b) / denom) if denom > 0 else 0.0

        def dot(a, b):
            return float(np.dot(a, b))

        def euclidean(a, b):
            return -float(np.linalg.norm(a - b))

        score_fn = {"cosine": cosine, "dot": dot, "euclidean": euclidean}[method]

        sims = []
        for uid, entry in self.index.items():
            if filter_fn and not filter_fn(entry["info"]):
                continue
            score = score_fn(query_vec, entry["embedding"])
            sims.append((uid, score))

        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]
