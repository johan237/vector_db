from sentence_transformers import SentenceTransformer


class EmbeddingFactory:
    MODELS = {
        "sentence-transformers": "all-MiniLM-L6-v2",
    }
    _cache = {}

    @classmethod
    def get_encoder(cls, embedding: str):
        if embedding not in cls.MODELS:
            raise ValueError(f"Unknown embedding method: {embedding}")
        if embedding not in cls._cache:
            print(f"Loading model for: {embedding}...")
            cls._cache[embedding] = SentenceTransformer(cls.MODELS[embedding])
        return cls._cache[embedding]

