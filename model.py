# ---------------- Vector DB Aggregator ----------------

from typing import Any, Dict, List, Tuple, Optional
from vector_db.storage.database import SQLiteDB
from vector_db.vector_dbs.product_vector_db import ProductVectorDB


class VectorDBAggregator:
    """
    Aggregator that unifies SQLite (storage) and multiple VectorDBs.
    It currently supports ProductVectorDB, but placeholders exist for
    ImageVectorDB and DocumentVectorDB.
    """

    def __init__(
        self,
        db_path: str = "products.db",
        embedding: str = "sentence-transformers",
        index_type: str = "brute",
    ):
        # Relational DB
        self.sqldb = SQLiteDB(db_path=db_path)

        # Vector DBs
        self.product_vecdb = ProductVectorDB(embedding=embedding, index_type=index_type)
        self.image_vecdb = None  # placeholder for future
        self.document_vecdb = None  # placeholder for future

    # ---------- Products ----------
    def add_product(self, type_: str, category: str, description: str) -> int:
        uid = self.sqldb.add_product(type_, category, description)
        info = {"type": type_, "category": category, "description": description}
        self.product_vecdb.add(uid, info)
        return uid

    def add_many_products(self, products: List[Dict[str, str]]) -> List[int]:
        uids = self.sqldb.add_many_products(products)
        uid_to_info = dict(zip(uids, products))
        self.product_vecdb.add_many(uid_to_info)
        return uids

    def search_products(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        method: str = "cosine",
    ) -> List[Tuple[Dict[str, Any], float]]:
        results = self.product_vecdb.search(query, top_k=top_k, filter=filter, method=method)
        enriched = []
        for uid, score in results:
            product = self.sqldb.get_product(uid)
            if product:
                enriched.append((product, score))
        return enriched

    def update_product(self, uid: int, type_: str = None, category: str = None, description: str = None) -> bool:
        product = self.sqldb.get_product(uid)
        if not product:
            return False

        new_type = type_ or product["type"]
        new_cat = category or product["category"]
        new_desc = description or product["description"]

        if not self.sqldb.update_product(uid, new_type, new_cat, new_desc):
            return False

        return self.product_vecdb.update(uid, {"type": new_type, "category": new_cat, "description": new_desc})

    def delete_product(self, uid: int) -> bool:
        if not self.sqldb.delete_product(uid):
            return False
        return self.product_vecdb.delete(uid)

    # ---------- Persistence ----------
    def save(self, vec_path: str = "vecdb.pkl"):
        """Save vector DBs to disk."""
        if self.product_vecdb:
            self.product_vecdb.save(vec_path)
        # self.image_vecdb.save(...)  (future)
        # self.document_vecdb.save(...)  (future)

    def load(self, vec_path: str = "vecdb.pkl"):
        """Load vector DBs from disk."""
        if self.product_vecdb:
            self.product_vecdb.load(vec_path)
        # self.image_vecdb.load(...)  (future)
        # self.document_vecdb.load(...)  (future)
