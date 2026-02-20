"""
RAG module over laws indices (bachelor_laws, master_laws).
Embeds query, retrieves from both collections, applies guardrails (relevance),
filters by similarity threshold, then generates answer with LLM (qwen3-32b). Refuses non-relevant queries.
"""
import logging
import os
import dotenv
dotenv.load_dotenv()
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Only include in context docs with similarity >= 40%
SIMILARITY_THRESHOLD = 0.4

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from llm_utils import chat_completion_with_thinking


EMBEDDING_MODEL = "BAAI/bge-m3"  # native sentence-transformers, Persian-friendly
QDRANT_HOST = os.environ.get("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION_BACHELOR = "bachelor_laws"
COLLECTION_MASTER = "master_laws"
LEVEL_LABELS = {
    COLLECTION_BACHELOR: "کارشناسی (bachelor)",
    COLLECTION_MASTER: "کارشناسی ارشد (master)",
}
TOP_K_PER_COLLECTION = 10
RAG_LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "qwen3-32b")

# Guardrail: is the query about university/educational regulations?
SYSTEM_GUARDRAIL = """You are a classifier. You must decide if the user's question is RELEVANT to university regulations, educational bylaws, or academic rules (آیین‌نامه آموزشی، قوانین دانشگاه، مقررات تحصیلی، کارشناسی، کارشناسی ارشد، واحد، نمره، مشروطی، و غیره).

Reply with exactly one word:
- RELEVANT: if the question is about such topics (even vaguely).
- IRRELEVANT: if the question is about something else (e.g. weather, sports, general knowledge, other institutions, or clearly off-topic).

Reply only RELEVANT or IRRELEVANT, nothing else."""

SYSTEM_ANSWER = """You are a helpful assistant that answers questions based ONLY on the provided context from university regulations (آیین‌نامه/قوانین آموزشی).

Rules:
- Answer in persian
- Use ONLY information from the context below. If the answer is not in the context, say so clearly.
- Do not perform any action (enrollment, deletion, etc.); only explain the rules.
- Be concise and cite the relevant part (e.g. ماده) when possible.
- If the context is empty or irrelevant, say you cannot answer from the available regulations.

When the context is split by degree level  (e.g. کارشناسی vs کارشناسی ارشد) and in the rules that are relevant to the question there are differences, you MUST answer separately for each case. Use clear headings such as:
- "If you are a bachelor student (کارشناسی), ..." / "اگر دانشجوی کارشناسی هستید، ..."
- "If you are a master student (کارشناسی ارشد), ..." / "اگر دانشجوی کارشناسی ارشد هستید، ..."
Only give one combined answer when the rules are the same for both levels; otherwise always separate by case."""

OUT_OF_SCOPE_MESSAGE = "این سؤال به آیین‌نامه و قوانین آموزشی دانشگاه مربوط نیست. لطفاً فقط در مورد مقررات تحصیلی، واحدها، نمرات، و مسائل مشابه بپرسید."


def _sanitize(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    return "".join(c if (ord(c) < 0xD800 or ord(c) > 0xDFFF) else "\ufffd" for c in s)


class RAGLawsHandler:
    def __init__(
        self,
        qdrant_host: str = QDRANT_HOST,
        qdrant_port: int = QDRANT_PORT,
        embedding_model_name: str = EMBEDDING_MODEL,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        model: Optional[str] = None,
        embedding_model: Optional[Any] = None,
    ):
        self._qdrant = QdrantClient(
            url=f"http://{qdrant_host}:{qdrant_port}",
            prefer_grpc=False,
            check_compatibility=False,
        )
        self._embedding_model = embedding_model or SentenceTransformer(embedding_model_name)
        self._client = OpenAI(
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=openai_base_url or os.environ.get("OPENAI_BASE_URL", "https://api.avalai.ir/v1"),
        )
        self._model = model or os.environ.get("RAG_LLM_MODEL", RAG_LLM_MODEL)

    def _embed(self, text: str) -> List[float]:
        return self._embedding_model.encode(text, show_progress_bar=False).tolist()

    def _retrieve(self, query: str, top_k: int = TOP_K_PER_COLLECTION) -> List[Dict[str, Any]]:
        """Retrieve from both bachelor and master collections; each result includes 'collection' and 'level_label'."""
        vector = self._embed(query)
        results = []
        for coll in (COLLECTION_BACHELOR, COLLECTION_MASTER):
            if not self._qdrant.collection_exists(coll):
                continue
            resp = self._qdrant.query_points(
                collection_name=coll,
                query=vector,
                limit=top_k,
            )
            points = getattr(resp, "points", None) or getattr(resp, "hits", None) or []
            for p in points:
                payload = getattr(p, "payload", None) or (p.get("payload") if isinstance(p, dict) else None)
                if payload:
                    score = getattr(p, "score", None)
                    if score is None and isinstance(p, dict):
                        score = p.get("score", 0.0)
                    if score is None:
                        score = 0.0
                    results.append({
                        **payload,
                        "collection": coll,
                        "level_label": LEVEL_LABELS.get(coll, coll),
                        "score": float(score),
                    })
        return results

    def _is_relevant(self, question: str) -> bool:
        """Guardrail: True only if question is about university/educational regulations."""
        try:
            content, _ = chat_completion_with_thinking(
                self._client,
                self._model,
                [
                    {"role": "system", "content": SYSTEM_GUARDRAIL},
                    {"role": "user", "content": _sanitize(question)},
                ],
                temperature=0.0,
            )
            return (content or "").strip().upper().startswith("RELEVANT")
        except Exception:
            return True  # on error, allow retrieval

    def answer(self, question: str) -> Tuple[str, bool]:
        """
        Returns (answer_text, was_answered).
        was_answered is False when the query was out of scope (guardrail blocked).
        """
        question = _sanitize((question or "").strip())
        if not question:
            return ("لطفاً یک سؤال بپرسید.", False)

        if not self._is_relevant(question):
            return (OUT_OF_SCOPE_MESSAGE, False)

        chunks = self._retrieve(question)
        # Log all retrieved docs with their similarity scores
        for i, c in enumerate(chunks, 1):
            logger.info(
                "RAG laws retrieved doc %d: collection=%s article_id=%s score=%.4f",
                i,
                c.get("collection", ""),
                c.get("article_id", ""),
                c.get("score", 0.0),
            )
        # Filter by similarity threshold (40%); only these go into context
        chunks = [c for c in chunks if (c.get("score") or 0) >= SIMILARITY_THRESHOLD]
        if chunks:
            logger.info(
                "RAG laws: keeping %d doc(s) above %.0f%% similarity for context",
                len(chunks),
                SIMILARITY_THRESHOLD * 100,
            )
        # Group by degree level so the LLM can answer per case when rules differ
        by_level: Dict[str, List[Dict[str, Any]]] = {}
        for c in chunks:
            label = c.get("level_label") or c.get("collection") or "other"
            by_level.setdefault(label, []).append(c)
        context_parts = []
        for level_label in (LEVEL_LABELS.get(COLLECTION_BACHELOR), LEVEL_LABELS.get(COLLECTION_MASTER)):
            level_chunks = by_level.get(level_label, [])
            if not level_chunks:
                continue
            block = "\n\n".join(
                f"[{c.get('article_id', '')}] {c.get('text', '')}" for c in level_chunks
            )
            context_parts.append(f"## {level_label}\n\n{block}")
        context = "\n\n---\n\n".join(context_parts) if context_parts else "\n\n---\n\n".join(
            f"[{c.get('article_id', '')}] {c.get('text', '')}" for c in chunks
        )
        if not context.strip():
            return ("در پایگاه مقررات آموزشی مطلبی مرتبط یافت نشد. لطفاً سؤال را دقیق‌تر کنید.", True)

        try:
            answer, thinking = chat_completion_with_thinking(
                self._client,
                self._model,
                [
                    {"role": "system", "content": SYSTEM_ANSWER},
                    {"role": "user", "content": f"متن مرجع:\n{_sanitize(context)}\n\nسؤال کاربر: {question}"},
                ],
                temperature=0.2,
            )
            if thinking:
                print(f"[thinking RAG answer]\n{thinking[:500]}...\n")
            return (_sanitize((answer or "").strip()), True)
        except Exception as e:
            return (f"خطا در تولید پاسخ: {e}", True)


def answer_question(
    question: str,
    handler: Optional[RAGLawsHandler] = None,
    **handler_kwargs,
) -> Tuple[str, bool]:
    """One-off or stateful. Returns (answer_text, was_answered)."""
    if handler is None:
        handler = RAGLawsHandler(**handler_kwargs)
    return handler.answer(question)


if __name__ == "__main__":
    h = RAGLawsHandler()
    while True:
        q = input("Question: ").strip()
        if not q:
            break
        out, answered = h.answer(q)
        print(f"Answered: {answered}\n{out}\n")
