from typing import Dict, Any, List, Tuple, Optional

from vector_db.storage.database import SQLiteDB
from vector_db.storage.schema import Schema
from vector_db.unifier.vector_manager import VectorStoreManager
from vector_db.vector_dbs.vector_db import VectorDB


class Aggregator:
    """
    Schema-driven aggregator that couples SQLite tables with their VectorDBs.
    """

    def __init__(self, db_path: str = "store.db"):
        self.db_path = db_path
        self.sql_dbs: Dict[str, SQLiteDB] = {}
        self.vec_manager = VectorStoreManager()

    # ---------------- Register new store ----------------
    def register(self, name: str, schema: Schema, vecdb: VectorDB):
        """
        Register a new domain (products, documents, images, ...).
        - Creates a SQLiteDB table using schema
        - Registers the corresponding VectorDB
        """
        sqldb = SQLiteDB(table_name=name, schema=schema, db_path=self.db_path)
        self.sql_dbs[name] = sqldb
        self.vec_manager.register(name, vecdb)

    # ---------------- CRUD ----------------
    def add_item(self, store: str, data: Dict[str, Any]) -> int:
        """Insert into SQLite + VectorDB."""
        uid = self.sql_dbs[store].add_item(data)
        self.vec_manager.get(store).add(uid, data)
        return uid

    def add_many_items(self, store: str, items: List[Dict[str, Any]]) -> List[int]:
        """Batch insert into SQLite + VectorDB."""
        uids = self.sql_dbs[store].add_many_items(items)
        self.vec_manager.get(store).add_many({uid: item for uid, item in zip(uids, items)})
        return uids

    def get_item(self, store: str, uid: int) -> Optional[Dict[str, Any]]:
        """Get a single item by ID (from SQLite)."""
        return self.sql_dbs[store].get_item(uid)

    def get_items(self, store: str) -> List[Dict[str, Any]]:
        """Get all items for a store (from SQLite)."""
        return self.sql_dbs[store].get_items()

    def update_item(self, store: str, uid: int, new_data: Dict[str, Any]) -> bool:
        """Update in SQLite + VectorDB."""
        ok = self.sql_dbs[store].update_item(uid, new_data)
        if ok:
            self.vec_manager.get(store).update(uid, new_data)
        return ok

    def delete_item(self, store: str, uid: int) -> bool:
        """Delete from SQLite + VectorDB."""
        ok = self.sql_dbs[store].delete_item(uid)
        if ok:
            self.vec_manager.get(store).delete(uid)
        return ok

    def delete_table(self, store: str) -> bool:
        """Drop a table + clear VectorDB."""
        ok = self.sql_dbs[store].delete_table()
        if ok:
            self.vec_manager.stores.pop(store, None)
        return ok

    # ---------------- Search ----------------
    def search_items(
        self,
        store: str,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        method: str = "cosine",
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search in a specific store (SQLite + VectorDB).
        Returns a list of (row, score).
        """
        results = self.vec_manager.search(query, store=store, top_k=top_k, filter=filter, method=method)
        enriched = []
        for uid, score in results.get(store, []):
            item = self.sql_dbs[store].get_item(uid)
            if item:
                enriched.append((item, score))
        return enriched

    def search_all(
        self, query: str, top_k: int = 5, method: str = "cosine"
    ) -> Dict[str, List[Tuple[Dict[str, Any], float]]]:
        """
        Search across all registered stores (products, documents, ...).
        Returns {store_name: [(row, score), ...]}
        """
        results = self.vec_manager.search(query, top_k=top_k, method=method)
        enriched = {}
        for store, matches in results.items():
            enriched[store] = []
            for uid, score in matches:
                item = self.sql_dbs[store].get_item(uid)
                if item:
                    enriched[store].append((item, score))
        return enriched
