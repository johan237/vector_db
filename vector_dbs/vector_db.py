from abc import ABC, abstractmethod
from typing import List, Any


class VectorDB(ABC):
    """
    Abstract base class for a vector database.
    Defines the required interface all vector DBs must implement.
    """

    @abstractmethod
    def add(self, uid: int, info: List[str: Any]):
        """Add an item with its uid and raw text (to be encoded into a vector)."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5):
        """Search for top_k most similar vectors to the query."""
        pass


"""
Implement all these search types:
- Flat index
- Inverted Flat Index 
- Product Quantization
- Hierarchical Navigable Small World (HNSW)

Recommended HNSW Parameters
Small datasets (≤10k vectors)

M = 16

ef_construction = 100

ef_search = 50
👉 fast, recall ~0.95, good enough.

Medium datasets (10k–100k vectors)

M = 32

ef_construction = 200–300

ef_search = 100–200
👉 balanced recall (0.95–0.99), query time still ~1–5ms.

Large datasets (100k–1M+ vectors)

M = 32–64

ef_construction = 300–500

ef_search = 200–400
👉 recall ~0.99 possible, query still ~1–10ms depending on hardware.
"""
