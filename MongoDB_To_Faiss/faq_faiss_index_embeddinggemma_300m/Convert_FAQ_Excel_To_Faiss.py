"""
Reads FAQ question/answer pairs from an Excel file and writes a dedicated
FAISS FAQ index to disk.

The Excel file is expected to have two columns: one for questions and one
for answers (see FAQ_QUESTION_COL / FAQ_ANSWER_COL below). Row 0 is treated
as a header and skipped automatically.

Each FAQ row is indexed as "سوال: <question> | پاسخ: <answer>" so semantic
search finds the right entry regardless of how the user phrases their query.

Run this whenever your FAQ Excel file changes.
"""

import json
from pathlib import Path

import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
EXCEL_FILE_PATH = "FAQ_Row_Wise.xlsx"   # Path to the FAQ Excel file
EXCEL_SHEET     = 0                     # Sheet index or name (0 = first sheet)

# Column names (or 0-based column indices) for question and answer.
# Change to integer indices (e.g. 0, 1) if your file has no header row
# and you set EXCEL_HEADER_ROW = None below.
FAQ_QUESTION_COL = "Question"
FAQ_ANSWER_COL   = "Answer"

# Row to use as column headers (0-indexed). Set to None if there is no header.
EXCEL_HEADER_ROW = 0

# Optional: a column that holds category / tag information (set to None to skip)
FAQ_CATEGORY_COL = None   # e.g. "Category"

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "embeddinggemma:300m"

# Output directory for the FAQ FAISS index
FAQ_INDEX_DIR = "faq_faiss_index"

SOURCE_NAME = "excel_faq"   # stored in metadata so downstream code knows the origin


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def clean(value) -> str:
    if value is None or (isinstance(value, float) and __import__("math").isnan(value)):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def create_faq_text(question: str, answer: str, category: str = "") -> str:
    """
    Build the searchable text that gets embedded.
    Format:  "سوال: <question> | پاسخ: <answer>"
    Pairing question and answer in one chunk improves semantic retrieval
    regardless of whether the user query matches the question or answer side.
    """
    parts = []
    if question:
        parts.append(f"سوال: {question}")
    if answer:
        parts.append(f"پاسخ: {answer[:600]}")   # cap answer length for embedding
    if category:
        parts.append(f"دسته: {category}")
    return " | ".join(parts) or "سوال متداول"


def build_faq_metadata(row_idx: int, question: str, answer: str, category: str = "") -> dict:
    """Store question + answer in metadata so views.py can retrieve them directly."""
    return {
        "collection": SOURCE_NAME,
        "question":   question,
        "answer":     answer,
        "category":   category,
        "_id":        str(row_idx),
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> bool:
    print("=" * 70)
    print("Excel  →  FAISS FAQ Index Converter")
    print("=" * 70)

    # 1. Load Excel
    excel_path = Path(EXCEL_FILE_PATH)
    print(f"\n1. Reading Excel file: {excel_path.resolve()}")
    if not excel_path.exists():
        print(f"   ✗ File not found: {excel_path}")
        return False

    try:
        df = pd.read_excel(
            excel_path,
            sheet_name=EXCEL_SHEET,
            header=EXCEL_HEADER_ROW,
            dtype=str,          # keep everything as text; avoids float coercion
        )
        print(f"   ✓ Loaded — {len(df)} rows, {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")
    except Exception as e:
        print(f"   ✗ Failed to read Excel: {e}")
        return False

    # Validate required columns
    for col in [FAQ_QUESTION_COL, FAQ_ANSWER_COL]:
        if col not in df.columns:
            print(f"   ✗ Column '{col}' not found. Available: {list(df.columns)}")
            return False

    # 2. Embeddings
    print(f"\n2. Loading embedding model: {EMBEDDING_MODEL}")
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        embeddings.embed_query("تست")
        print("   ✓ Embedding model ready")
    except Exception as e:
        print(f"   ✗ Ollama error: {e}")
        return False

    # 3. Build documents — each row is one chunk
    print("\n3. Processing FAQ rows…")
    documents: list[Document] = []
    skipped = 0

    for idx, row in df.iterrows():
        try:
            question = clean(row[FAQ_QUESTION_COL])
            answer   = clean(row[FAQ_ANSWER_COL])
            category = clean(row[FAQ_CATEGORY_COL]) if FAQ_CATEGORY_COL and FAQ_CATEGORY_COL in df.columns else ""

            if not question and not answer:
                skipped += 1
                continue

            text = create_faq_text(question, answer, category)
            meta = build_faq_metadata(idx, question, answer, category)
            documents.append(Document(page_content=text, metadata=meta))

        except Exception as e:
            skipped += 1
            print(f"\n   ⚠ Skipped row {idx}: {str(e)[:100]}")

    print(f"   Total FAQ documents: {len(documents)}  (skipped: {skipped})")
    if not documents:
        print("   ✗ Nothing to index.")
        return False

    # 4. Sample: show what will be indexed
    print("\n   Sample indexed texts:")
    for doc in documents[:3]:
        print(f"   • {doc.page_content[:120]}")

    # 5. Build & save FAISS index
    print("\n4. Building FAISS FAQ index…")
    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
        print("   ✓ Index created")
    except Exception as e:
        print(f"   ✗ FAISS build failed: {e}")
        return False

    print(f"\n5. Saving to: {FAQ_INDEX_DIR}/")
    try:
        out = Path(FAQ_INDEX_DIR)
        out.mkdir(exist_ok=True)
        vectorstore.save_local(str(out))
        print(f"   ✓ Saved to '{FAQ_INDEX_DIR}/'")
    except Exception as e:
        print(f"   ✗ Save failed: {e}")
        return False

    # 6. Smoke test — sample up to 5 questions evenly from the loaded Excel data
    print("\n6. Smoke tests…")
    all_questions = [clean(q) for q in df[FAQ_QUESTION_COL] if clean(q)]
    step = max(1, len(all_questions) // 5)
    test_queries = all_questions[::step][:5]
    for q in test_queries:
        try:
            results = vectorstore.similarity_search(q, k=2)
            print(f"\n   Query: '{q}'")
            for r in results:
                print(f"      Q: {r.metadata.get('question', '?')[:70]}")
                print(f"      A: {r.metadata.get('answer', '?')[:70]}")
        except Exception as e:
            print(f"   ⚠ '{q}' failed: {e}")

    print("\n" + "=" * 70)
    print(f"✓  FAQ INDEX COMPLETE  —  {len(documents)} vectors  →  {FAQ_INDEX_DIR}/")
    print("=" * 70)
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)