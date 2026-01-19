# ---------------- Product Vector DB ----------------
import time
import threading
from typing import Any, Dict, Callable, List, Tuple, Union
import numpy as np
import pickle

from vector_db.embedding import EmbeddingFactory
from vector_db.vector_dbs.vector_db import VectorDB

try:
    import hnswlib
    HAS_HNSWLIB = True
except ImportError:
    HAS_HNSWLIB = False


class ProductVectorDB(VectorDB):
    """
    Vector DB specialized for products.
    Stores {uid: {"embedding": vector, "info": dict}}.
    Supports brute-force search or scalable ANN via HNSW.
    """

    def __init__(
        self,
        embedding: str = "sentence-transformers",
        index_type: str = "brute",
        hnsw_M: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 50,
    ):
        self.encoder = EmbeddingFactory.get_encoder(embedding)
        self.index: Dict[int, Dict[str, Any]] = {}
        self.index_type = index_type
        self.dim = None

        # HNSW parameters
        self.hnsw = None
        self.hnsw_M = hnsw_M
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search

        if index_type == "hnsw" and not HAS_HNSWLIB:
            raise ImportError("hnswlib is required for HNSW. Install it with `pip install hnswlib`.")

        # Registry of scoring functions
        self.scoring_functions: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
            "cosine": self._cosine,
            "dot": self._dot,
            "euclidean": self._euclidean,
        }

        # Lock for thread safety
        self.lock = threading.Lock()

    # ---------- Internal scoring functions ----------
    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # normalized on add

    def _dot(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # same as cosine if normalized

    def _euclidean(self, a: np.ndarray, b: np.ndarray) -> float:
        return -float(np.linalg.norm(a - b))  # higher = closer

    # ---------- Core methods ----------
    def add(self, uid: int, info: Dict[str, Any]):
        """Add a single product by wrapping add_many."""
        self.add_many({uid: info})

    def add_many(self, items: Dict[int, Dict[str, Any]]):
        """
        Batch insert: encode all items at once for speed.
        items: {uid: info}
        """
        with self.lock:
            texts = []
            uids = []
            for uid, info in items.items():
                text_parts = [str(v) for v in info.values() if isinstance(v, str)]
                texts.append(" ".join(text_parts))
                uids.append(uid)

            # Batch encode and normalize
            vecs = self.encoder.encode(texts).astype(np.float32)
            vecs = np.array([self._normalize(v) for v in vecs])

            # Insert into in-memory index
            for uid, vec, info in zip(uids, vecs, items.values()):
                self.index[uid] = {"embedding": vec, "info": info}

            # If HNSW is enabled, add them there too
            if self.index_type == "hnsw":
                if self.hnsw is None:
                    self.dim = vecs.shape[1]
                    self.hnsw = hnswlib.Index(space="cosine", dim=self.dim)
                    self.hnsw.init_index(
                        max_elements=100000,
                        ef_construction=self.hnsw_ef_construction,
                        M=self.hnsw_M,
                    )
                    self.hnsw.set_ef(self.hnsw_ef_search)
                self.hnsw.add_items(vecs, uids)

    def update(self, uid: int, new_info: Dict[str, Any]) -> bool:
        """
        Update an item in the vector index.
        """
        if uid not in self.index:
            return False

        text_parts = [str(v) for v in new_info.values() if isinstance(v, str)]
        text = " ".join(text_parts)
        vec = self.encoder.encode([text])[0].astype(np.float32)
        vec = self._normalize(vec)

        self.index[uid] = {"embedding": vec, "info": new_info}
        return True

    def delete(self, uid: int) -> bool:
        with self.lock:
            if uid in self.index:
                del self.index[uid]
                # hnswlib does not support true deletion (rebuild needed)
                return True
            return False

    def search(
            self,
            query: str,
            top_k: int = 5,
            filter: Union[Dict[str, Any], Callable[[Dict[str, Any]], bool], None] = None,
            method: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "cosine",
    ) -> List[Tuple[int, float]]:
        """
        Perform similarity search.
        Returns list of (uid, score).

        filter:
          - dict: {"key": value, ...} → exact match
          - callable: function(info) -> bool → custom filter
        method: "cosine", "dot", "Euclidean", or custom callable(a, b).
        """
        if not self.index:
            return []

        # Resolve scoring function
        if isinstance(method, str):
            if method not in self.scoring_functions:
                raise ValueError(f"Unknown scoring method '{method}'")
            score_fn = self.scoring_functions[method]
        else:
            score_fn = method

        # Build filter function
        if isinstance(filter, dict):
            def filter_fn(info: Dict[str, Any]) -> bool:
                return all(info.get(k) == v for k, v in filter.items())
        elif callable(filter):
            filter_fn = filter
        else:
            filter_fn = None

        query_vec = self.encoder.encode([query])[0].astype(np.float32)
        query_vec = self._normalize(query_vec)
        sims = []

        if self.index_type == "hnsw" and self.hnsw is not None:
            labels, distances = self.hnsw.knn_query(query_vec, k=top_k * 2)
            for uid, dist in zip(labels[0], distances[0]):
                if uid not in self.index:
                    continue
                info = self.index[uid]["info"]
                if filter_fn and not filter_fn(info):
                    continue
                score = 1 - dist
                sims.append((uid, float(score)))
        else:
            for uid, entry in self.index.items():
                if filter_fn and not filter_fn(entry["info"]):
                    continue
                vec = entry["embedding"]
                score = score_fn(query_vec, vec)
                sims.append((uid, score))

        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    # ---------- Persistence ----------
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.index, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.index = pickle.load(f)

    # ---------- Benchmark ----------
    def benchmark(self, num_vectors=50000, dim=384, num_queries=10, top_k=10):
        if not HAS_HNSWLIB:
            raise ImportError("hnswlib not installed. Run `pip install hnswlib`")

        data = np.random.randn(num_vectors, dim).astype(np.float32)
        data /= np.linalg.norm(data, axis=1, keepdims=True)

        brute_index = {i: {"embedding": v, "info": {}} for i, v in enumerate(data)}

        hnsw = hnswlib.Index(space="cosine", dim=dim)
        start = time.time()
        hnsw.init_index(
            max_elements=1200000,
            ef_construction=self.hnsw_ef_construction,
            M=self.hnsw_M,
        )

        hnsw.add_items(data, list(range(num_vectors)))
        hnsw.set_ef(self.hnsw_ef_search)  # <--- ADD IT HERE

        build_time = time.time() - start

        queries = np.random.randn(num_queries, dim).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        start = time.time()
        brute_results = []
        for q in queries:
            sims = [(i, float(np.dot(q, entry['embedding']))) for i, entry in brute_index.items()]
            sims.sort(key=lambda x: x[1], reverse=True)
            brute_results.append([uid for uid, _ in sims[:top_k]])
        brute_time = (time.time() - start) / num_queries

        start = time.time()
        hnsw_results = []
        for q in queries:
            labels, distances = hnsw.knn_query(q, k=top_k)
            hnsw_results.append(labels[0].tolist())
        hnsw_time = (time.time() - start) / num_queries

        recalls = []
        for b, h in zip(brute_results, hnsw_results):
            overlap = len(set(b) & set(h)) / len(b)
            recalls.append(overlap)
        avg_recall = np.mean(recalls)

        return {
            "num_vectors": num_vectors,
            "dim": dim,
            "build_time_hnsw": build_time,
            "avg_query_time_brute_ms": brute_time * 1000,
            "avg_query_time_hnsw_ms": hnsw_time * 1000,
            "recall": avg_recall,
        }
