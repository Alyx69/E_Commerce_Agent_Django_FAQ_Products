"""
Reads every MongoDB collection and writes a dedicated FAISS product index to disk.
Run this whenever your product data changes.
"""

import json
import re
from pathlib import Path

from bson import ObjectId
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pymongo import MongoClient

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MONGO_URI       = "mongodb://localhost:27017"
DATABASE_NAME   = "product_store"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "embeddinggemma:300m"

# Output directory for the product FAISS index
PRODUCT_INDEX_DIR = "products_faiss_index"

# The FAQ collection name — it will be skipped here
FAQ_COLLECTION_NAME = "faq"

# Collections to skip in addition to the FAQ collection
EXTRA_SKIP_COLLECTIONS: set[str] = set()   # e.g. {"logs", "system.users"}

# ── Field name mappings (Persian + English) ──────────────────────────────────
TITLE_FIELDS = [
    "title",
    "name",
    "product_name",
    "عنوان",
    "نام",
]

DESC_FIELDS = [
    "description",
    "desc",
    "توضیحات",
    "شرح",
]

CATEGORY_FIELDS = [
    "category",
    "دسته‌بندی",
    "دسته",
]

PRICE_FIELDS = [
    "price",
    "current_price",
    "قیمت",
]

PRICE_HISTORY_FIELDS = [
    "price_history",
]

RATING_FIELDS = [
    "rating",
    "ratings",
    "امتیاز",
    "add_to_love(ratings)",
]

URL_FIELDS = [
    "url",
    "link",
    "لینک",
]

WARRANTY_FIELDS = [
    "warranty",
    "گارانتی",
    "specifications.ضمانت",
]

AVAILABLE_FIELDS = [
    "available",
    "in_stock",
    "موجود",
]

STORE_FIELDS = [
    "store_name",
    "platform",
    "فروشگاه",
]

COLOR_FIELDS = [
    "color",
    "رنگ",
    "specifications.رنگ",
]

BRAND_FIELDS = [
    "brand",
    "برند",
]

# ─────────────────────────────────────────────
# Helpers (shared with FAQ converter)
# ─────────────────────────────────────────────

def _get(doc: dict, fields: list[str], default=""):
    for f in fields:
        v = doc.get(f)
        if v is not None and v != "" and v != "نامشخص":
            return v
        if "." in f:
            parts = f.split(".")
            cur = doc
            for p in parts:
                m = re.match(r"^(.*)\[(\d+)\]$", p)
                if m:
                    key, idx = m.group(1), int(m.group(2))
                    cur = cur.get(key, [])
                    cur = cur[idx] if isinstance(cur, list) and len(cur) > idx else {}
                else:
                    cur = cur.get(p, {}) if isinstance(cur, dict) else {}
            if cur and cur != {}:
                return cur
    return default


def clean(value) -> str:
    if value is None or value == "" or value == "نامشخص":
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def format_price(raw) -> str:
    try:
        num = float(str(raw).replace(",", ""))
        return f"{int(num):,} تومان"
    except Exception:
        return "قیمت نامشخص"


def get_latest_price(doc: dict) -> str:
    """
    Extract the most recent price from a product document.
    Strategy (in order):
      1. price_history[-1].current_price  — preferred (most stores use this)
      2. Top-level price fields (price, current_price, قیمت, …)
    Returns a formatted string like "399,000 تومان" or "قیمت نامشخص".
    """
    # 1. Walk price_history — take the LAST entry's current_price
    history = doc.get("price_history")
    if isinstance(history, list) and history:
        last_entry = history[-1]
        if isinstance(last_entry, dict):
            raw = last_entry.get("current_price")
            if raw not in (None, "", "نامشخص"):
                return format_price(raw)

    # 2. Fallback: top-level price fields
    raw = _get(doc, PRICE_FIELDS)
    if raw not in (None, "", "نامشخص", {}):
        return format_price(raw)

    return "قیمت نامشخص"


def serialize_doc(doc: dict) -> dict:
    out = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            out[k] = str(v)
        elif isinstance(v, dict):
            out[k] = serialize_doc(v)
        elif isinstance(v, list):
            out[k] = [serialize_doc(i) if isinstance(i, dict) else
                      (str(i) if isinstance(i, ObjectId) else i)
                      for i in v]
        else:
            out[k] = v
    return out


def create_product_text(doc: dict) -> str:
    """Build rich searchable text for a product document."""
    parts = []
    title = clean(_get(doc, TITLE_FIELDS))
    if title:
        parts.append(title)
    desc = clean(_get(doc, DESC_FIELDS))
    if desc and desc != "بدون توضیحات":
        parts.append(desc[:500])
    category = clean(_get(doc, CATEGORY_FIELDS))
    if category:
        parts.append(f"دسته‌بندی: {category}")
    return " | ".join(p for p in parts if p) or "محصول"


def build_product_metadata(doc: dict, collection_name: str) -> dict:
    price_str = get_latest_price(doc)

    rating = clean(_get(doc, RATING_FIELDS)) or "بدون امتیاز"

    available_raw = _get(doc, AVAILABLE_FIELDS)
    if isinstance(available_raw, bool):
        available_str = "موجود" if available_raw else "ناموجود"
    else:
        available_str = clean(available_raw) or "نامشخص"

    meta = {
        "collection": collection_name,
        "platform":   clean(_get(doc, STORE_FIELDS)) or collection_name,
        "title":      clean(_get(doc, TITLE_FIELDS)) or "محصول",
        "url":        clean(_get(doc, URL_FIELDS))   or "#",
        "price":      price_str,
        "rating":     rating,
        "category":   clean(_get(doc, CATEGORY_FIELDS)) or "نامشخص",
        "available":  available_str,
        "warranty":   clean(_get(doc, WARRANTY_FIELDS)) or "گارانتی نامشخص",
        "_id":        "",  # filled below
    }
    color = clean(_get(doc, COLOR_FIELDS))
    if color:
        meta["color"] = color
    return meta


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> bool:
    print("=" * 70)
    print("MongoDB  →  FAISS Product Index Converter")
    print("=" * 70)

    SKIP_COLLECTIONS = EXTRA_SKIP_COLLECTIONS | {FAQ_COLLECTION_NAME}

    # 1. Connect
    print(f"\n1. Connecting to MongoDB: {MONGO_URI}")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5_000)
        client.server_info()
        db = client[DATABASE_NAME]
        print(f"   ✓ Connected — database: '{DATABASE_NAME}'")
    except Exception as e:
        print(f"   ✗ MongoDB connection failed: {e}")
        return False

    # 2. Discover collections (skip FAQ)
    all_collections = db.list_collection_names()
    collections = [c for c in all_collections if c not in SKIP_COLLECTIONS]
    print(f"\n2. Product collections to index: {collections}")
    print(f"   (Skipping: {SKIP_COLLECTIONS & set(all_collections)})")
    if not collections:
        print("   ✗ No product collections found.")
        return False

    # 3. Embeddings
    print(f"\n3. Loading embedding model: {EMBEDDING_MODEL}")
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        embeddings.embed_query("تست")
        print("   ✓ Embedding model ready")
    except Exception as e:
        print(f"   ✗ Ollama error: {e}")
        return False

    # 4. Build documents
    print("\n4. Processing product documents…")
    documents: list[Document] = []
    skipped = 0

    for coll_name in collections:
        collection = db[coll_name]
        total = collection.count_documents({})
        print(f"\n   Collection: '{coll_name}'  ({total} docs)")
        for idx, raw_doc in enumerate(collection.find()):
            try:
                doc  = serialize_doc(raw_doc)
                text = create_product_text(doc)
                meta = build_product_metadata(doc, coll_name)
                meta["_id"] = str(raw_doc.get("_id", ""))
                documents.append(Document(page_content=text, metadata=meta))
                if (idx + 1) % 100 == 0:
                    print(f"      … {idx + 1}/{total}", end="\r")
            except Exception as e:
                skipped += 1
                print(f"\n      ⚠ Skipped doc {idx}: {str(e)[:100]}")
        print(f"      ✓ Done with '{coll_name}'")

    print(f"\n   Total product documents: {len(documents)}  (skipped: {skipped})")
    if not documents:
        print("   ✗ Nothing to index.")
        return False

    # 5. Build & save FAISS index
    print("\n5. Building FAISS product index…")
    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
        print("   ✓ Index created")
    except Exception as e:
        print(f"   ✗ FAISS build failed: {e}")
        return False

    print(f"\n6. Saving to: {PRODUCT_INDEX_DIR}/")
    try:
        out = Path(PRODUCT_INDEX_DIR)
        out.mkdir(exist_ok=True)
        vectorstore.save_local(str(out))
        print(f"   ✓ Saved to '{PRODUCT_INDEX_DIR}/'")
    except Exception as e:
        print(f"   ✗ Save failed: {e}")
        return False

    # 7. Smoke test
    print("\n7. Smoke tests…")
    for q in ["هدفون", "شارژر", "لپ‌تاپ"]:
        try:
            results = vectorstore.similarity_search(q, k=2)
            print(f"   '{q}' → {[r.metadata.get('title','?')[:40] for r in results]}")
        except Exception as e:
            print(f"   ⚠ '{q}' failed: {e}")

    print("\n" + "=" * 70)
    print(f"✓  PRODUCT INDEX COMPLETE  —  {len(documents)} vectors  →  {PRODUCT_INDEX_DIR}/")
    print("=" * 70)
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)