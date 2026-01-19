# from vector_db.storage.database import SQLiteDB
#
# from vector_dbs.product_vector_db import ProductVectorDB
#
# if __name__ == "__main__":
#     # Init DBs
#     sqldb = SQLiteDB()
#     vdb = ProductVectorDB(
#         embedding="sentence-transformers",
#         index_type="brute",
#         hnsw_M=48,
#         hnsw_ef_construction=500,
#         hnsw_ef_search=500,
#     )
#
#     # Add products with flexible schema
#     vdb.add(1, {"type": "electronics", "category": "phone", "description": "A smartphone with good camera"})
#     vdb.add(2, {"type": "electronics", "category": "laptop", "description": "A laptop with long battery life"})
#     vdb.add(3, {"type": "grocery", "category": "fruit", "description": "Fresh organic bananas"})
#
#     # Normal semantic search
#     print("\nSearch: 'phone with camera'")
#     results = vdb.search("phone with camera", top_k=2)
#     print(results)
#
#     # Search with filter by category
#     print("\nSearch: 'something sweet', filter category=fruit")
#     results = vdb.search("something sweet", filter=lambda info: info.get("category") == "fruit")
#     print(results)
#
#     report = vdb.benchmark(num_vectors=200000, dim=384, num_queries=100, top_k=20)
#     print(report)
from sentence_transformers import SentenceTransformer

from vector_db.storage.schema import Schema
from vector_db.unifier.aggregator import Aggregator
from vector_db.vector_dbs.product_vector_db import ProductVectorDB
from vector_db.vector_dbs.vector_schema import SchemaVectorDB
from utils.doc_reader import build_doc_item
if __name__ == "__main__":
    '''
    # ---------------- Define Schema ----------------
    PRODUCT_SCHEMA = Schema(
        required={"type": str, "category": str, "description": str},
        optional={"price": float, "brand": str}
    )

    # ---------------- Init Aggregator ----------------
    agg = Aggregator(db_path="store.db")

    # Register products store
    agg.register("products", PRODUCT_SCHEMA, ProductVectorDB())

    # ---------------- Add Products ----------------
    print("\n--- Adding products ---")
    uid1 = agg.add_item("products", {
        "type": "electronics",
        "category": "phone",
        "description": "A smartphone with an excellent camera",
        "brand": "TechCo",
        "price": 699.99
    })
    uid2 = agg.add_item("products", {
        "type": "electronics",
        "category": "laptop",
        "description": "A laptop with long battery life",
        "brand": "NotePro",
        "price": 1199.00
    })

    uids_batch = agg.add_many_items("products", [
        {"type": "grocery", "category": "fruit", "description": "Fresh organic bananas", "price": 2.99},
        {"type": "grocery", "category": "fruit", "description": "Juicy red apples", "price": 3.49}
    ])

    print(f"Added products: {uid1}, {uid2}, {uids_batch}")

    # ---------------- Get Items ----------------
    print("\n--- Get single product ---")
    print(agg.get_item("products", uid1))

    print("\n--- Get all products ---")
    print(agg.get_items("products"))

    # ---------------- Search ----------------
    print("\n--- Search for 'phone with camera' ---")
    results = agg.search_items("products", "phone with camera", top_k=3)
    for r in results:
        print(r)

    print("\n--- Search for 'fruit' ---")
    results = agg.search_items("products", "fruit", top_k=3)
    for r in results:
        print(r)

    # ---------------- Update ----------------
    print("\n--- Update product ---")
    agg.update_item("products", uid1, {"description": "A smartphone with dual cameras"})
    print(agg.get_item("products", uid1))

    # ---------------- Delete ----------------
    print("\n--- Delete product ---")
    agg.delete_item("products", uid2)
    print(agg.get_items("products"))

    # ---------------- Save & Load ----------------
    print("\n--- Save vector DB ---")
    agg.save("vecdb.pkl")

    print("\n--- Load vector DB ---")
    agg.load("vecdb.pkl")

    print("\n--- Cross-domain search (only products for now) ---")
    results = agg.search_all("smartphone")
    print(results)
'''
    from sentence_transformers import SentenceTransformer
    from utils.doc_reader import build_doc_item

    # ------------------- Define models -------------------
    # Small, lightweight model for titles/descriptions
    small_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim

    # Larger, more accurate model for content chunks
    big_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")  # 768-dim

    # ------------------- Define schema -------------------
    DOC_SCHEMA = {
        "embed_fields": [
            {
                "fields": ["title", "description"],
                "embedding_model": small_model,  # lightweight
                "dim": 384,
                "chunked": False
            },
            {
                "fields": ["content"],
                "embedding_model": big_model,  # heavier, more accurate
                "dim": 768,
                "chunked": True,
                "chunk_size": 250,
                "overlap": 50
            }
        ],
        "filters": ["author"]
    }

    # Init schema-based vector DB
    db = SchemaVectorDB(DOC_SCHEMA)

    # ------------------- Add a document -------------------
    item = build_doc_item(
        uid=1,
        title="A Fuzzy Logic-Based Adaptive Framework for Context-Aware and Safe Cooking Assistance with COOK",
        description=(
            "Abstract. This research enhances COOK (Cognitive Orthosis for cOoKing) by introducing "
            "a fuzzy logic-based adaptive framework overcoming static rule-based limitations, enabling "
            "context-aware cooking assistance adapting to individual cognitive impairments. "
            "We developed a four context fuzzy inference system integrating stove operations, "
            "environmental conditions, cooking methods, and user profiles to replace COOK’s static rules. "
            "Dynamic coefficient adaptation personalizes risk assessment based on user cognitive profiles, "
            "with weights auto-adjusted using incident history and temporal factors."
        ),
        path=r"C:\Users\johan\OneDrive\Desktop\Data Science\NLP\COOK__Fuzzy.pdf",
        author="Johan"
    )

    # Add it to DB
    db.add(item["uid"], item)

    # ------------------- Queries -------------------
    print("\n🔍 Search for 'Stove Context (Cs)' inside document chunks:")
    results1 = db.search_in_document(1, "Stove Context (Cs)", field_group=1)
    for r in results1:
        print(f"Page={r['page']} | Start={r['start']} End={r['end']} | Score={r['score']:.3f}")
        print(f"   Text: {r['text'][:100]}...\n")

    print("\n🔍 Search for 'R2 = 0.759' inside document chunks:")
    results2 = db.search_in_document(1, "R2 = 0.759", field_group=1)
    for r in results2:
        print(f"Page={r['page']} | Start={r['start']} End={r['end']} | Score={r['score']:.3f}")
        print(f"   Text: {r['text'][:100]}...\n")
