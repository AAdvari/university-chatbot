"""
Clarification module for the chatbot: handles ambiguous questions by asking
clarifying questions via LLM, then returns a unified clear question when done.
"""
import os
import re
import dotenv
dotenv.load_dotenv()
from typing import Tuple, List, Dict, Any, Optional

from ambiguity_detector import AmbiguityDetector


def _sanitize_unicode(s: str) -> str:
    """Remove surrogate characters so text is safe for JSON/UTF-8 (e.g. API requests)."""
    if not s or not isinstance(s, str):
        return s or ""
    return "".join(
        c if (ord(c) < 0xD800 or ord(c) > 0xDFFF) else "\ufffd"
        for c in s
    )
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from llm_utils import chat_completion_with_thinking


# --- Config (override via env or constructor) ---
COLLECTION_NAME = "ambiguous_questions"
# EMBEDDING_MODEL = "multilingual-e5-large"  # native sentence-transformers, Persian-friendly
EMBEDDING_MODEL = "BAAI/bge-m3"  # native sentence-transformers, Persian-friendly
QDRANT_HOST = os.environ.get("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
TOP_K_RETRIEVAL = int(os.environ.get("TOP_K_RETRIEVAL", "5"))  # top-K candidates for LLM to choose from

# LLM: generate clarifying questions from top-K candidates; stay close to given questions, minimal creativity
SYSTEM_GENERATE_CLARIFICATIONS = """You are a helpful assistant. You are given:
1. The user's ambiguous question (in Persian).
2. A list of similar candidate questions (in Persian), each with example clarifying questions that were used for that candidate.

Your task: Produce a short list of clarifying questions (in Persian) to disambiguate the user's question.

Important: Stay as close as possible to the provided clarifying questions. Prefer using them exactly as written, or with only minimal wording changes if necessary. Do not invent new questions or rephrase creatively—choose and reuse the given questions so they match the user's intent. Minimize unnecessary creativity.

Output format: List each clarifying question on its own line, numbered (1. ... 2. ... 3. ...). Output only this numbered list, no other text. Typically 2 to 5 questions."""

# LLM instructions for next clarifying question vs all answered
SYSTEM_CLARIFY = """You are a helpful assistant that asks clarifying questions in Persian.
You are given:
1. The dialogue so far (user and assistant messages).
2. A list of clarifying questions that we need answers for.

Your task:
- If the user indicates they **do not want to answer** a clarification question or **do not know** the answer (e.g. نمیدانم، مهم نیست، هرکدام): **ignore that question**—do not ask it again. Skip it and either ask the next unanswered clarification from the list, or if no more are needed, respond with "ALL_ANSWERED:" and a unified question.
- If not all clarifying questions have been answered yet (and the user has not said to skip): ask the user for the next unanswered clarification. Use the wording of that clarification from the list as much as possible. Respond with ONLY that single question (Persian), no prefix or label.
- If the dialogue already contains answers that cover all the clarifying points, or the user skipped/did not specify some: respond with exactly the line "ALL_ANSWERED:" (in English) followed by a single rephrased, clear, unified question in Persian that incorporates the user's answers and reflects that any skipped question was ignored (user did not specify that aspect).

Important: When the user does not want to answer or does not know, ignore that question and move on; do not insist.

Output format:
- Either: one clarifying question (Persian), preferably verbatim from the list.
- Or: ALL_ANSWERED: <one unified clear question in Persian>"""

SYSTEM_REPHRASE = """You are a helpful assistant. You are given the full dialogue so far: the user's original question and every clarifying question from the assistant with the user's answers.

Your task: Using ALL the information in this dialogue, write a single clear, unambiguous question in Persian that fully expresses what the user wants to know. The question must incorporate the original intent plus every clarification the user provided. If the user did not want to answer a clarification or said they do not know, **ignore that question**—do not require an answer. For those points, the unified question may note that the user did not specify (e.g. "بدون مشخص بودن مقطع" or "با فرض هر حالت") so the system can still answer. Output only that one question, nothing else."""


class ClarificationHandler:
    """
    Stateful handler: process(user_text) -> (response_text, clarification_done).
    One instance per conversation/session.
    """

    def __init__(
        self,
        ambiguity_model_dir: str = "./saved_ambiguity_model",
        qdrant_host: str = QDRANT_HOST,
        qdrant_port: int = QDRANT_PORT,
        collection_name: str = COLLECTION_NAME,
        embedding_model_name: str = EMBEDDING_MODEL,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        embedding_model: Any = None,
        ambiguity_detector: Any = None,
    ):
        self._ambiguity_detector = ambiguity_detector or AmbiguityDetector(ambiguity_model_dir)
        qdrant_url = f"http://{qdrant_host}:{qdrant_port}"
        self._qdrant = QdrantClient(
            url=qdrant_url, prefer_grpc=False, check_compatibility=False
        )
        self._collection = collection_name
        self._embedding_model = embedding_model or SentenceTransformer(embedding_model_name)
        self._client = OpenAI(
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=openai_base_url or os.environ.get("OPENAI_BASE_URL", "https://api.avalai.ir/v1"),
        )
        self._model = os.environ.get("OPENAI_CHAT_MODEL", "qwen3-8b")

        # Per-session state
        self._dialogue: List[Dict[str, str]] = []
        self._clarifications: List[str] = []
        self._in_clarification = False

    def _embed(self, text: str) -> List[float]:
        return self._embedding_model.encode(text, show_progress_bar=False).tolist()

    def _search_similar_top_k(self, question: str, k: int) -> List[Dict[str, Any]]:
        """Return top-k similar questions from Qdrant, each with payload (question, clarifications)."""
        vector = self._embed(question)
        response = self._qdrant.query_points(
            collection_name=self._collection,
            query=vector,
            limit=k,
        )
        points = getattr(response, "points", None) or getattr(response, "hits", None) or []
        result = []
        for p in points:
            payload = getattr(p, "payload", None) if hasattr(p, "payload") else (p.get("payload") if isinstance(p, dict) else None)
            if payload and payload.get("clarifications"):
                result.append(payload)
        return result

    def _call_llm_generate_clarifications(self, user_question: str, candidates: List[Dict[str, Any]]) -> List[str]:
        """Ask LLM to generate clarifying questions from the user's question and top-K candidates. Returns list of questions."""
        if not candidates:
            return []
        lines = [
            f"User's question: {_sanitize_unicode(user_question)}",
            "",
            "Similar candidate questions and their clarifying questions (use as inspiration; you may use as-is, adapt, or mix):",
        ]
        for i, c in enumerate(candidates, 1):
            q = c.get("question", "")
            cl = c.get("clarifications", [])
            lines.append(f"Candidate {i} - Question: {_sanitize_unicode(q)}")
            for x in cl:
                lines.append(f"  - {_sanitize_unicode(x)}")
        content, thinking = chat_completion_with_thinking(
            self._client,
            self._model,
            [
                {"role": "system", "content": SYSTEM_GENERATE_CLARIFICATIONS},
                {"role": "user", "content": "\n".join(lines)},
            ],
            temperature=0.01,
        )
        if thinking:
            print(f"[thinking generate_clarifications]\n{thinking[:500]}...\n")
        return self._parse_numbered_clarifications(content, candidates)

    def _parse_numbered_clarifications(self, content: str, fallback_candidates: List[Dict[str, Any]]) -> List[str]:
        """Parse LLM output into list of clarifying questions (numbered 1. 2. ... or similar). Fallback to first candidate's list if empty."""
        result = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            # Remove leading "N." or "N)" or "- " (e.g. "1. سوال؟" or "2) سوال؟" or "- سوال؟")
            rest = re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
            rest = rest.lstrip("-").strip()
            if rest:
                result.append(_sanitize_unicode(rest))
        if not result and fallback_candidates:
            result = list(fallback_candidates[0].get("clarifications") or [])
        return result

    def _llm_messages(self, dialogue: List[Dict[str, str]], clarifications: List[str]) -> List[Dict[str, str]]:
        lines = [
            "Dialogue so far:",
            *[f"{m['role'].upper()}: {_sanitize_unicode(m['content'])}" for m in dialogue],
            "",
            "Clarifying questions we need answers for:",
            *[f"- {_sanitize_unicode(c)}" for c in clarifications],
        ]
        body = _sanitize_unicode("\n".join(lines))
        return [
            {"role": "system", "content": SYSTEM_CLARIFY},
            {"role": "user", "content": body},
        ]

    def _call_llm_clarify(self, dialogue: List[Dict[str, str]], clarifications: List[str]) -> str:
        content, thinking = chat_completion_with_thinking(
            self._client,
            self._model,
            self._llm_messages(dialogue, clarifications),
        )
        if thinking:
            print(f"[thinking clarify]\n{thinking[:500]}...\n")
        return _sanitize_unicode(content)

    def _call_llm_rephrase(self, dialogue: List[Dict[str, str]]) -> str:
        text = "\n".join([f"{m['role'].upper()}: {_sanitize_unicode(m['content'])}" for m in dialogue])
        content, thinking = chat_completion_with_thinking(
            self._client,
            self._model,
            [
                {"role": "system", "content": SYSTEM_REPHRASE},
                {"role": "user", "content": text},
            ],
            temperature=0.01,
        )
        if thinking:
            print(f"[thinking rephrase]\n{thinking[:500]}...\n")
        return _sanitize_unicode(content)

    def _print_dialogue_state(self, round_name: str = "") -> None:
        """Print current dialogue state after each round (for debugging duplicate questions)."""
        print(f"\n--- Dialogue state {round_name} ---")
        print(f"  _in_clarification: {self._in_clarification}")
        print(f"  _dialogue length: {len(self._dialogue)}")
        for i, m in enumerate(self._dialogue):
            role = m.get("role", "?")
            content = (m.get("content", "") or "")[:80]
            print(f"    [{i}] {role}: {content!r}...")
        print(f"  _clarifications ({len(self._clarifications)}): {[c[:50] + '...' if len(c) > 50 else c for c in self._clarifications]}")
        print("---\n")

    def process(self, user_text: str) -> Tuple[str, bool]:
        """
        Process one user message. Returns (response_text, clarification_done).
        - clarification_done=True: no more clarifying needed; response_text is either the
          original question (if not ambiguous) or the unified rephrased question.
        - clarification_done=False: we are still in clarification; response_text is the
          next clarifying question to show the user.
        """
        user_text = _sanitize_unicode((user_text or "").strip())
        if not user_text:
            return ("لطفاً یک سؤال یا پیام وارد کنید.", True)

        # Already in clarification flow: append user reply and ask LLM for next step
        if self._in_clarification:
            self._dialogue.append({"role": "user", "content": user_text})
            reply = self._call_llm_clarify(self._dialogue, self._clarifications)

            if reply.upper().startswith("ALL_ANSWERED:"):
                # Always rephrase full dialogue into one clear question using all given information
                unified = self._call_llm_rephrase(self._dialogue)
                if not unified or not unified.strip():
                    unified = reply.split("ALL_ANSWERED:", 1)[-1].strip() or user_text
                self._print_dialogue_state("(after ALL_ANSWERED, before reset)")
                self._reset_state()
                return (unified, True)

            self._dialogue.append({"role": "assistant", "content": reply})
            self._print_dialogue_state("(in clarification, after assistant reply)")
            return (reply, False)

        # New turn: check ambiguity
        pred = self._ambiguity_detector.predict(user_text)
        if pred != "Ambiguous":
            return (user_text, True)

        # Ambiguous: get top-K similar questions from Qdrant, LLM generates clarifying questions from them
        candidates = self._search_similar_top_k(user_text, TOP_K_RETRIEVAL)
        if not candidates:
            # No similar ambiguous question in index: treat as clear enough
            return (user_text, True)

        self._clarifications = self._call_llm_generate_clarifications(user_text, candidates)
        if not self._clarifications:
            return (user_text, True)
        self._in_clarification = True
        self._dialogue = [
            {"role": "user", "content": user_text},
        ]

        # Get first clarifying question from LLM
        first_question = self._call_llm_clarify(self._dialogue, self._clarifications)
        if first_question.upper().startswith("ALL_ANSWERED:"):
            # Use full dialogue to produce one clear question (only one exchange so far)
            unified = self._call_llm_rephrase(self._dialogue)
            if not unified or not unified.strip():
                unified = first_question.split("ALL_ANSWERED:", 1)[-1].strip() or user_text
            self._print_dialogue_state("(ambiguous first turn ALL_ANSWERED, before reset)")
            self._reset_state()
            return (unified, True)

        self._dialogue.append({"role": "assistant", "content": first_question})
        self._print_dialogue_state("(ambiguous first turn, after first question)")
        return (first_question, False)

    def _reset_state(self) -> None:
        self._dialogue = []
        self._clarifications = []
        self._in_clarification = False

    def reset_session(self) -> None:
        """Call when starting a new conversation."""
        self._reset_state()


def process_text(
    user_text: str,
    handler: Optional[ClarificationHandler] = None,
    **handler_kwargs,
) -> Tuple[str, bool]:
    """
    One-off or stateful processing. If handler is None, a new handler is created
    (no state across calls). Pass the same handler for a multi-turn conversation.
    Returns (response_text, clarification_done).
    """
    if handler is None:
        handler = ClarificationHandler(**handler_kwargs)
    return handler.process(user_text)


if __name__ == "__main__":
    # Minimal demo (requires Qdrant running, OPENAI_API_KEY, and built index)
    h = ClarificationHandler()
    while True:
        text = input("User: ").strip()
        print(text)
        if not text:
            break
        out, done = h.process(text)
        print(f"Clarification done: {done}\nResponse: {out}\n")
        if done:
            h.reset_session()
