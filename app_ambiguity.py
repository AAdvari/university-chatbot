"""
Doc-based ambiguity detection and clarification flow for university regulations.
Uses retrieved documents + LLM to decide if a question is ambiguous and to generate
clarification questions and reformulated questions.
"""
import json
import re
from dataclasses import dataclass, field


@dataclass
class RoundResult:
    question: str
    docs: list[dict]
    is_ambiguous: bool
    reason: str
    clarification_question: str | None = None
    answer: str | None = None
    round_num: int = 0


@dataclass
class SessionState:
    phase: str = "idle"
    current_question: str = ""
    conversation_history: list[dict] = field(default_factory=list)
    round_num: int = 0
    pending_clarification: str = ""
    pending_docs: list[dict] = field(default_factory=list)
    pending_reason: str = ""

    def reset(self):
        self.phase = "idle"
        self.current_question = ""
        self.conversation_history = []
        self.round_num = 0
        self.pending_clarification = ""
        self.pending_docs = []
        self.pending_reason = ""


AMBIGUITY_DETECTION_SYSTEM = (
    "تو یک دستیار هوشمند هستی که در تحلیل آیین‌نامه‌های دانشگاهی تخصص داری.\n"
    "وظیفه تو این است که مشخص کنی آیا سوال کاربر، با توجه به اسناد بازیابی‌شده، مبهم است یا خیر.\n\n"
    "معیار ابهام (مهم):\n"
    "- اگر **همه اسناد بازیابی‌شده به سوال کاربر نامرتبط هستند یا سندی بازیابی نشده است**، سوال را **واضح** (غیرمبهم) در نظر بگیر و reason بگذار مثلاً «اسناد بازیابی‌شده به سوال مرتبط نیستند؛ رفع ابهام لازم نیست.»\n\n"
    "- سوال را **واضح** در نظر بگیر اگر همه اسناد بازیابی‌شده که مرتبط به سوال هستند به یک پاسخ درست منجر شوند، یا کاربر قبلاً مشخص کرده "
    "کدام سند/شرایط (مثلاً مقطع تحصیلی) مد نظرش است.\n"
    "- سوال را **مبهم** در نظر بگیر اگر تکیه کردن به اسناد مختلفی که مرتبط به سوال هستند (اسناد غیرمرتبط را درنظر نگیر)، پاسخ درست سیستم را عوض کند.\n"
    "پاسخ خود را دقیقاً به فرمت JSON زیر بده و هیچ متن اضافی ننویس:\n"
    '{"is_ambiguous": true/false, "reason": "دلیل به فارسی"}'
)

CLARIFICATION_SYSTEM = (
    "تو یک دستیار هوشمند هستی که در تحلیل آیین‌نامه‌های دانشگاهی تخصص داری.\n"
    "اسناد بازیابی‌شده برای سوال کاربر، پاسخ‌های متفاوتی می‌دهند؛ بنابراین باید مشخص شود کدام سند/کدام زمینه "
    "برای پاسخ‌گویی معتبر است.\n\n"
    "وظیفه تو: یک سوال رفع ابهام به فارسی بنویس که به کاربر کمک کند مشخص کند **کدام سند (یا کدام آیین‌نامه/مقطع) "
    "باید مبنای پاسخ قرار بگیرد**. مثلاً: «آیا منظور شما مقطع کارشناسی است یا کارشناسی ارشد؟» یا «کدام آیین‌نامه "
    "مد نظر شماست؟»\n\n"
    "سوال رفع ابهام باید:\n"
    "- کوتاه و واضح باشد\n"
    "- گزینه‌های ممکن (مثلاً اسناد یا مقاطع مختلف) را مشخص کند\n"
    "- به فارسی باشد\n"
    "- فقط درباره چیزی باشد که هنوز روشن نشده (سوال تکراری نپرس)\n\n"
    "پاسخ خود را دقیقاً به فرمت JSON زیر بده و هیچ متن اضافی ننویس:\n"
    '{"clarification_question": "سوال رفع ابهام به فارسی"}'
)

REFORMULATE_SYSTEM = (
    "تو یک دستیار هوشمند هستی. وظیفه تو این است که با توجه به سوال اولیه کاربر و پاسخ او "
    "به سوال رفع ابهام، یک سوال جدید و دقیق‌تر بسازی.\n\n"
    "سوال جدید باید:\n"
    "- تمام اطلاعات سوال اولیه و پاسخ کاربر را در خود داشته باشد\n"
    "- واضح و بدون ابهام باشد\n"
    "- به فارسی باشد\n\n"
    "پاسخ خود را دقیقاً به فرمت JSON زیر بده و هیچ متن اضافی ننویس:\n"
    '{"reformulated_question": "سوال جدید به فارسی"}'
)


def parse_json_from_llm(text: str) -> dict:
    """Extract JSON object from LLM response text."""
    text = text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        text = json_match.group(0)
    return json.loads(text)


def format_documents(docs: list[dict]) -> str:
    """Format retrieved docs for LLM context."""
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(
            f"--- سند {i} (منبع: آیین‌نامه {d['source']}، ماده {d['article_number']}) ---\n"
            f"{d['text']}"
        )
    return "\n\n".join(parts)


def format_conversation_history(history: list[dict]) -> str:
    """Format conversation history for LLM context."""
    if not history:
        return ""
    parts = []
    for h in history:
        parts.append(
            f"- سوال قبلی: {h['question']}\n"
            f"  سوال رفع ابهام: {h['clarification']}\n"
            f"  پاسخ کاربر: {h['user_answer']}"
        )
    return "تاریخچه مکالمه (ابهام‌هایی که قبلاً رفع شده‌اند):\n" + "\n".join(parts)


def filter_docs_by_question(question: str, docs: list[dict]) -> list[dict]:
    """Filter docs by degree level mentioned in question (کارشناسی / ارشد)."""
    if "کارشناسی ارشد" in question or "ارشد" in question:
        filtered = [d for d in docs if d["source"] == "کارشناسی ارشد"]
        return filtered if filtered else docs
    if "کارشناسی" in question or "کاردانی" in question:
        filtered = [d for d in docs if d["source"] == "کارشناسی"]
        return filtered if filtered else docs
    return docs


# Similarity threshold for document retrieval: keep only docs with score >= this (40%)
SIMILARITY_THRESHOLD = 0.4

# LLM: decide if the user's reply to a clarification question means "don't know" / "refuse to specify"
REFUSAL_DETECTION_SYSTEM = """You are a classifier. You are given:
1. A clarification question that was asked (in Persian).
2. The user's reply (in Persian).

Decide whether the user's reply indicates that they **do not know**, **do not want to specify**, or that **any option is fine** (e.g. "I don't know", "doesn't matter", "either way", "هرکدام", "نمیدانم", "مهم نیست"). If yes, output REFUSES.
If the user gave a clear, specific answer (e.g. chose one option like "کارشناسی" or "ارشد"), output ANSWERS.

Reply with exactly one word: REFUSES or ANSWERS. No other text."""


def filter_docs_by_similarity(docs: list[dict], min_score: float | None = None) -> list[dict]:
    """Filter out documents with similarity score lower than min_score (default 40%)."""
    threshold = min_score if min_score is not None else SIMILARITY_THRESHOLD
    return [d for d in docs if (d.get("score") or 0) >= threshold]


def no_relevant_docs(docs: list[dict], min_score: float | None = None) -> bool:
    """
    True if we should skip doc-based disambiguation and pass the question directly
    to the next stage (e.g. RAG laws): no docs retrieved or all scores below min_score.
    Uses SIMILARITY_THRESHOLD (40%) when min_score is not given.
    """
    threshold = min_score if min_score is not None else SIMILARITY_THRESHOLD
    if not docs:
        return True
    return all((d.get("score") or 0) < threshold for d in docs)


class DocDisambiguation:
    """
    Doc-based ambiguity detection: given a question and retrieved docs,
    detect ambiguity, generate clarification question, or reformulate question.
    """

    def __init__(self, llm_client, llm_model: str = "qwen3-8b"):
        self._llm = llm_client
        self.llm_model = llm_model

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        response = self._llm.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            stream=False,
            extra_body={"enable_thinking": False},
        )
        return response.choices[0].message.content

    def detect_ambiguity(
        self,
        question: str,
        docs: list[dict],
        conversation_history: list[dict] | None = None,
    ) -> dict:
        """Return {"is_ambiguous": bool, "reason": str}."""
        context = format_documents(docs)
        history_text = format_conversation_history(conversation_history or [])
        prompt = (
            f"سوال کاربر: {question}\n\n"
            f"{history_text}\n\n"
            f"اسناد بازیابی‌شده:\n{context}\n\n"
            "آیا تکیه کردن به اسناد مختلف (مثلاً سند ۱ در مقابل سند ۲) پاسخ درست سیستم را عوض می‌کند؟ "
            "اگر بله، سوال مبهم است؛ اگر خیر (همه اسناد به یک پاسخ منجر می‌شوند یا زمینه از قبل روشن است)، واضح است."
        )
        response = self._call_llm(AMBIGUITY_DETECTION_SYSTEM, prompt)
        try:
            return parse_json_from_llm(response)
        except (json.JSONDecodeError, ValueError):
            is_ambiguous = "true" in response.lower() or "مبهم" in response
            return {"is_ambiguous": is_ambiguous, "reason": response}

    def generate_clarification(
        self,
        question: str,
        docs: list[dict],
        reason: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        """Return a single clarification question (Persian)."""
        context = format_documents(docs)
        history_text = format_conversation_history(conversation_history or [])
        prompt = (
            f"سوال کاربر: {question}\n\n"
            f"دلیل ابهام: {reason}\n\n"
            f"{history_text}\n\n"
            f"اسناد بازیابی‌شده:\n{context}\n\n"
            "یک سوال رفع ابهام بنویس که به کاربر کمک کند مشخص کند کدام سند/کدام آیین‌نامه (یا مقطع) باید مبنای پاسخ باشد. قبلاً پرسیده نشده باشد."
        )
        response = self._call_llm(CLARIFICATION_SYSTEM, prompt)
        try:
            parsed = parse_json_from_llm(response)
            return parsed.get("clarification_question", response)
        except (json.JSONDecodeError, ValueError):
            return response

    def reformulate_question(
        self,
        original_question: str,
        clarification_q: str,
        user_answer: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        """Return reformulated question (Persian) incorporating user's answer."""
        history_text = format_conversation_history(conversation_history or [])
        prompt = (
            f"سوال اولیه: {original_question}\n"
            f"سوال رفع ابهام: {clarification_q}\n"
            f"پاسخ کاربر: {user_answer}\n\n"
            f"{history_text}\n\n"
            "لطفاً یک سوال جدید و دقیق‌تر بنویس که تمام اطلاعات مشخص‌شده توسط کاربر "
            "(مثلاً مقطع تحصیلی) را صریحاً شامل شود."
        )
        response = self._call_llm(REFORMULATE_SYSTEM, prompt)
        try:
            parsed = parse_json_from_llm(response)
            return parsed.get("reformulated_question", response)
        except (json.JSONDecodeError, ValueError):
            return response

    def interprets_as_refusal(self, clarification_question: str, user_answer: str) -> bool:
        """Use LLM to decide if the user's reply means they don't know or refuse to specify (treat as clarified, do not ask again)."""
        if not (user_answer or "").strip():
            return False
        prompt = (
            f"سوال رفع ابهام: {clarification_question}\n\n"
            f"پاسخ کاربر: {user_answer.strip()}"
        )
        response = (self._call_llm(REFUSAL_DETECTION_SYSTEM, prompt) or "").strip().upper()
        return "REFUSES" in response
