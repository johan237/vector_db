import os
import pickle
from typing import Dict, Union, List, Optional


class VectorStoreManager:
    def __init__(self):
        self.stores: Dict[str, object] = {}

    # -------- Registry --------
    def register(self, name: str, store):
        """Register a vector store under a given name."""
        self.stores[name] = store

    def unregister(self, name: str):
        """Remove a vector store from manager."""
        if name in self.stores:
            del self.stores[name]

    def get(self, name: str):
        """Retrieve a vector store by name."""
        return self.stores.get(name)

    def list_stores(self) -> List[str]:
        """List all registered store names."""
        return list(self.stores.keys())

    # -------- Search --------
    def search(
            self,
            query: str,
            store: Union[str, List[str]],
            top_k: int = 5,
            **kwargs
    ):
        """Search one or multiple stores."""
        if isinstance(store, str):
            store = [store]

        results = {}
        for s in store:
            if s not in self.stores:
                continue
            results[s] = self.stores[s].search(query, top_k=top_k, **kwargs)
        return results

    # -------- Persistence --------
    def save_all(self, base_path: str = "vector_stores"):
        """Save all stores to disk."""
        os.makedirs(base_path, exist_ok=True)
        for name, store in self.stores.items():
            path = os.path.join(base_path, f"{name}.pkl")
            store.save(path)

    def load_all(self, base_path: str = "vector_stores"):
        """Load all stores from disk."""
        for name, store in self.stores.items():
            path = os.path.join(base_path, f"{name}.pkl")
            if os.path.exists(path):
                store.load(path)
