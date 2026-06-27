import asyncio
import json
import re
from pathlib import Path
from typing import Literal

from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.decorators.http import require_http_methods

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage

from config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    PRODUCTS_INDEX_DIR,
    FAQ_INDEX_DIR,
)

# ─────────────────────────────────────────────────────────────────────────────
# Model initialisation
# ─────────────────────────────────────────────────────────────────────────────

# JSON-mode model — product formatting only
model_json = ChatOllama(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    base_url=OLLAMA_BASE_URL,
    format="json",
)

# Plain-text model — FAQ answers, greetings, intent classification
model_text = ChatOllama(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    base_url=OLLAMA_BASE_URL,
)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


# Shared identity / behaviour rules injected into BOTH prompts
_SHARED_RULES = """Shared rules:
1. Always respond in Persian (Farsi), regardless of the language the user writes in.
2. Base every answer strictly on data retrieved from the vector database. Never guess, infer, or fabricate information.
3. If the user greets you or exchanges pleasantries, reply politely and warmly in Persian.
4. Keep all responses clear, polite, and appropriate in tone."""

# ── FAQ system prompt ────────────────────────────────────────────────────────
FAQ_SYSTEM_PROMPT = f"""You are an intelligent support assistant for a payment application and online store.
Your job is to answer users' frequently asked questions using information retrieved from the vector database.

{_SHARED_RULES}

FAQ-specific rules:
5. Your answers must be based entirely on the retrieved FAQ entries provided to you. Do not add any information from outside that context.
6. If the retrieved entries do not contain enough information to fully answer the question, honestly say that you do not know the complete answer.
7. Do not respond to questions that are entirely unrelated to payment application support or the online store. Politely redirect the user instead.
8. Write your response as fluent Persian prose — no JSON, no code blocks, no markdown."""

# ── Product system prompt ────────────────────────────────────────────────────
PRODUCT_SYSTEM_PROMPT = f"""You are a product search assistant. You receive retrieved product results from a database and format them for the user.

{_SHARED_RULES}

Product search-specific rules:
Return results in exactly the JSON structure below — no extra text, explanation, or emojis:
{{
  "platforms": [
    {{
      "platform": "Platform name",
      "results": [
        {{
          "url": "product URL",
          "title": "product title"
        }}
      ]
    }}
  ]
}}
Include only the products present in the retrieved results provided to you. Never invent or guess URLs.
If a platform has no results, omit it from the JSON entirely."""


# ─────────────────────────────────────────────────────────────────────────────
# Index paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR           = Path(__file__).parent.parent
PRODUCT_INDEX_PATH = BASE_DIR / PRODUCTS_INDEX_DIR
FAQ_INDEX_PATH     = BASE_DIR / FAQ_INDEX_DIR


# ─────────────────────────────────────────────────────────────────────────────
# FAISS loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_index(path: Path) -> FAISS | None:
    try:
        if not path.exists():
            raise FileNotFoundError(f"Index not found at {path}")
        return FAISS.load_local(
            str(path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        print(f"Error loading FAISS index at {path}: {e}")
        return None


def load_product_index() -> FAISS | None:
    return _load_index(PRODUCT_INDEX_PATH)


def load_faq_index() -> FAISS | None:
    return _load_index(FAQ_INDEX_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Product search
# ─────────────────────────────────────────────────────────────────────────────

def search_products(query: str, top_k: int = 30, max_distance: float = 1.2) -> dict:
    vectorstore = load_product_index()
    if vectorstore is None:
        return {"error": "ایندکس محصولات در دسترس نیست"}
    try:
        results = vectorstore.similarity_search_with_score(query, k=top_k)
        grouped: dict[str, list] = {}
        for doc, distance in results:
            if distance > max_distance:
                continue
            meta     = doc.metadata
            platform = meta.get("platform", "نامشخص")
            if platform not in grouped:
                grouped[platform] = []
            grouped[platform].append({
                "title":            meta.get("title", "بدون عنوان"),
                "url":              meta.get("url", "#"),
                "price":            meta.get("price", "قیمت نامشخص"),
                "description":      doc.page_content,
                "similarity_score": float(1.0 / (1.0 + distance)),
                "distance":         float(distance),
            })
        for platform in grouped:
            grouped[platform].sort(key=lambda x: x["distance"])
        return grouped
    except Exception as e:
        print(f"Product search error: {e}")
        return {"error": str(e)}


def get_all_products_from_faiss() -> list[dict]:
    vectorstore = load_product_index()
    if vectorstore is None:
        return []
    try:
        all_docs = vectorstore.similarity_search("", k=1000)
        products = []
        for doc in all_docs:
            meta = doc.metadata
            products.append({
                "platform":    meta.get("platform", "نامشخص"),
                "title":       meta.get("title", "بدون عنوان"),
                "url":         meta.get("url", "#"),
                "price":       meta.get("price", "قیمت نامشخص"),
                "description": doc.page_content,
            })
        products.sort(key=lambda x: (x["platform"], x["title"]))
        return products
    except Exception as e:
        print(f"Error retrieving all products: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# FAQ search
# ─────────────────────────────────────────────────────────────────────────────

def search_faq(query: str, top_k: int = 5, max_distance: float = 1.5) -> list[dict]:
    vectorstore = load_faq_index()
    if vectorstore is None:
        return []
    try:
        results = vectorstore.similarity_search_with_score(query, k=top_k * 3)
        hits: list[dict] = []
        for doc, distance in results:
            if distance > max_distance:
                continue
            meta     = doc.metadata
            question = meta.get("question", "")
            answer   = meta.get("answer", "")
            if not question and not answer:
                lines    = doc.page_content.split("|", 1)
                question = lines[0].replace("سوال:", "").strip() if lines else doc.page_content
                answer   = lines[1].replace("پاسخ:", "").strip() if len(lines) > 1 else ""
            hits.append({
                "question":    question,
                "answer":      answer,
                "category":    meta.get("category", ""),
                "distance":    float(distance),
                "raw_content": doc.page_content,
            })
            if len(hits) >= top_k:
                break
        return sorted(hits, key=lambda x: x["distance"])
    except Exception as e:
        print(f"FAQ search error: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Intent classification
# ─────────────────────────────────────────────────────────────────────────────

# Greeting detection — handled before FAISS search
_GREETING_PATTERNS = [
    "سلام", "درود", "خوبی", "چطوری", "حالت", "صبح بخیر", "عصر بخیر",
    "شب بخیر", "hello", "hi ", "hey", "good morning", "good evening",
]


def _is_greeting(query: str) -> bool:
    q = query.lower().strip()
    return any(g in q for g in _GREETING_PATTERNS) and len(q.split()) <= 6


async def classify_intent(query: str) -> Literal["faq", "product", "greeting"]:
    """
    Classify query as greeting, faq, or product.
    Heuristic-first; LLM fallback only on ambiguous cases.
    """
    if _is_greeting(query):
        return "greeting"

    q = query.lower()

    faq_signals = [
        "چطور", "چگونه", "چیست", "چی هست", "چه ", "کجا", "آیا",
        "مرجوع", "برگشت", "ارسال", "تحویل", "گارانتی", "پشتیبانی",
        "پرداخت", "کنسل", "لغو", "رمز", "حساب", "ثبت‌نام", "ورود",
        "سامین", "ساختی", "کی ساخت",
        "how", "what", "why", "when", "where", "return", "refund",
        "shipping", "warranty", "support", "cancel", "policy", "?", "؟",
    ]
    product_signals = [
        "بخر", "خرید", "قیمت", "ارزان", "بهترین", "محصول", "کالا",
        "لپ‌تاپ", "گوشی", "هدفون", "شارژر", "کابل", "تلویزیون", "مانیتور",
        "buy", "shop", "cheap", "best", "price", "laptop", "phone",
        "headphone", "cable", "charger",
    ]

    faq_score     = sum(1 for kw in faq_signals     if kw in q)
    product_score = sum(1 for kw in product_signals if kw in q)

    if faq_score > product_score:
        return "faq"
    if product_score > faq_score:
        return "product"

    # Ambiguous — ask the LLM using the FAQ system prompt for consistent identity
    classification_prompt = f"""The user entered the following message:
"{query}"

Classify it into exactly one of these three categories:
- faq      → a support, informational, or application-related question
- product  → searching for a product to buy or compare
- greeting → a greeting, farewell, or casual pleasantry

Reply with exactly one word: faq or product or greeting"""

    try:
        response = await model_text.ainvoke([
            SystemMessage(content=FAQ_SYSTEM_PROMPT),
            HumanMessage(content=classification_prompt),
        ])
        content = (response.content if hasattr(response, "content") else str(response)).strip().lower()
        if "greeting" in content:
            return "greeting"
        if "faq" in content:
            return "faq"
    except Exception as e:
        print(f"Intent classification error: {e}")

    return "product"  # safe default


# ─────────────────────────────────────────────────────────────────────────────
# Greeting agent
# ─────────────────────────────────────────────────────────────────────────────

async def run_greeting_agent(query: str) -> dict:
    """
    Handles greetings using the FAQ system prompt so identity rules apply
    (language, Samin attribution, tone) without touching either index.
    """
    try:
        response = await model_text.ainvoke([
            SystemMessage(content=FAQ_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ])
        answer = (response.content if hasattr(response, "content") else str(response)).strip()
    except Exception as e:
        print(f"Greeting agent error: {e}")
        answer = "سلام! خوش آمدید. چطور می‌توانم کمکتان کنم؟"  # greeting answer stays Persian per system prompt rules

    return {
        "type":    "faq",          # renders in the FAQ card UI
        "answer":  answer,
        "sources": [],
        "message": "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# FAQ agent
# ─────────────────────────────────────────────────────────────────────────────

async def run_faq_agent(query: str) -> dict:
    hits = search_faq(query, top_k=5)

    if not hits:
        return {
            "type":    "faq",
            "answer":  None,
            "sources": [],
            "message": f'پاسخی برای «{query}» یافت نشد. لطفاً سوال خود را با عبارت دیگری بیان کنید.',
        }

    # Build retrieved context
    context_parts = []
    for i, hit in enumerate(hits, 1):
        if hit["answer"]:
            context_parts.append(f"Entry {i} — Question: {hit['question']}\nEntry {i} — Answer: {hit['answer']}")
        else:
            context_parts.append(f"Entry {i}: {hit['raw_content']}")
    context = "\n\n".join(context_parts)

    # The system prompt already sets all identity + behaviour rules.
    # The human message provides only the task-specific context.
    user_message = f"""User question: {query}

Relevant entries retrieved from the database:
{context}

Write an appropriate answer in Persian:"""

    try:
        response = await model_text.ainvoke([
            SystemMessage(content=FAQ_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])
        answer = (response.content if hasattr(response, "content") else str(response)).strip()
    except Exception as e:
        print(f"FAQ LLM error: {e}")
        best   = hits[0]
        answer = best["answer"] or best["raw_content"]

    return {
        "type":    "faq",
        "answer":  answer,
        "sources": [{"question": h["question"], "answer": h["answer"]} for h in hits],
        "message": f'پاسخ برای «{query}» یافت شد',
    }


# ─────────────────────────────────────────────────────────────────────────────
# Product agent
# ─────────────────────────────────────────────────────────────────────────────

async def run_product_agent(query: str) -> dict:
    search_results = search_products(query, top_k=30)

    if "error" in search_results:
        return {"type": "product", "platforms": [], "message": "خطا در بارگذاری ایندکس محصولات."}

    if not search_results or all(not v for v in search_results.values()):
        return {
            "type":      "product",
            "platforms": [],
            "message":   f'محصولی برای «{query}» یافت نشد. لطفاً عبارت دیگری جستجو کنید.',
        }

    # Build retrieval summary
    summary_lines = []
    total_results = 0
    for platform, products in search_results.items():
        top = products[:5]
        total_results += len(top)
        summary_lines.append(f"\n{platform} ({len(top)} products):")
        for i, p in enumerate(top, 1):
            summary_lines.append(f"  {i}. {p['title']} | {p['price']}")

    if total_results == 0:
        return {
            "type":      "product",
            "platforms": [],
            "message":   f'محصولی برای «{query}» یافت نشد.',
        }

    search_text = "\n".join(summary_lines)

    # System prompt carries all rules; human message is pure task data.
    user_message = f"""Search query: "{query}"
Total retrieved results: {total_results}

Results:
{search_text}"""

    try:
        response = await model_json.ainvoke([
            SystemMessage(content=PRODUCT_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])
        content = (response.content if hasattr(response, "content") else str(response)).strip()
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        parsed = json.loads(content)

        if "platforms" in parsed and isinstance(parsed["platforms"], list):
            # Re-attach real URLs from FAISS (LLM must not invent them)
            for block in parsed["platforms"]:
                pname = block.get("platform", "")
                if pname in search_results:
                    originals = search_results[pname]
                    for i, result in enumerate(block.get("results", [])):
                        if i < len(originals):
                            result["url"]         = originals[i]["url"]
                            result["description"] = originals[i].get("description", "")
                            result["price"]       = originals[i].get("price", "")

            parsed["type"]          = "product"
            parsed["total_results"] = total_results
            parsed["message"]       = f'{total_results} محصول برای «{query}» یافت شد'
            return parsed

    except Exception as e:
        print(f"Product LLM parse error: {e}")

    # Fallback: build directly from FAISS results
    platforms = []
    for platform, products in search_results.items():
        if products:
            platforms.append({
                "platform": platform,
                "results": [
                    {"url": p["url"], "title": p["title"], "description": p.get("description", ""), "price": p.get("price", "")}
                    for p in products[:5]
                ],
            })

    return {
        "type":          "product",
        "platforms":     platforms,
        "total_results": total_results,
        "message":       f'{total_results} محصول برای «{query}» یافت شد',
    }


# ─────────────────────────────────────────────────────────────────────────────
# Unified agent entry point
# ─────────────────────────────────────────────────────────────────────────────

async def run_agent(query: str) -> dict:
    intent = await classify_intent(query)
    print(f"[agent] query='{query}'  intent={intent}")
    if intent == "greeting":
        return await run_greeting_agent(query)
    if intent == "faq":
        return await run_faq_agent(query)
    return await run_product_agent(query)


# ─────────────────────────────────────────────────────────────────────────────
# Django view
# ─────────────────────────────────────────────────────────────────────────────

@require_http_methods(["GET", "POST"])
def index(request):
    all_products = get_all_products_from_faiss()

    if request.method == "POST":
        query = request.POST.get("query", "").strip()
        if not query:
            messages.error(request, "لطفاً یک عبارت جستجو وارد کنید.")
            return redirect("index")

        try:
            response_json = asyncio.run(run_agent(query))
        except Exception as exc:
            messages.error(request, f"خطای سیستم: {exc}")
            return redirect("index")

        return render(request, "index.html", {
            "query":        query,
            "response":     response_json,
            "all_products": all_products,
        })

    return render(request, "index.html", {
        "query":        "",
        "response":     None,
        "all_products": all_products,
    })