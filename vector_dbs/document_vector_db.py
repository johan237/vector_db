from vector_db.embedding import EmbeddingFactory
from vector_db.vector_dbs.vector_db import VectorDB


class DocumentVectorDB(VectorDB):
    """
    Vector DB specialized for documents.
    Stores {chunk_id: {"embedding": vec, "doc_id": int, "chunk": str}}
    Maps doc_id -> metadata separately.
    """

    def __init__(self, embedding: str = "sentence-transformers", index_type: str = "brute"):
        self.encoder = EmbeddingFactory.get_encoder(embedding)
        self.index = {}   # chunk_id -> {"embedding": vec, "doc_id": id, "chunk": str}
        self.docs = {}    # doc_id -> {"title": ..., "author": ..., "created_at": ...}
        self.index_type = index_type
        self.next_chunk_id = 0


        """
        Methods it should have: 
        - add
        - search
        - add_document
        - update_document
        - delete_document
        - search
        - search_in_document : Uses chunks to reference the chunk thus page numbers which reference the best a particular query
        """

