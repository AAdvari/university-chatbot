from __future__ import annotations

import os
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

# Pipeline: user question → clarification_module → app doc-based disambiguation → clarified_qa (RAG laws) → answer
from clarification_module import ClarificationHandler
from rag_laws_module import RAGLawsHandler
from app_ambiguity import (
    RoundResult,
    SessionState,
    DocDisambiguation,
    format_documents,
    filter_docs_by_question,
    filter_docs_by_similarity,
    no_relevant_docs,
)

# Same Qdrant as clarification_module (app doc-based flow uses this for university_regulations)
QDRANT_HOST = os.environ.get("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))


# ---------------------------------------------------------------------------
# DisambiguationRAG  --  pipeline-ready class (indexing + ambiguity in separate modules)
# ---------------------------------------------------------------------------


class DisambiguationRAG:

    COLLECTION = "university_regulations"

    ANSWER_SYSTEM = (
        "تو یک دستیار هوشمند هستی که در تحلیل آیین‌نامه‌های دانشگاهی تخصص داری.\n"
        "با توجه به اسناد بازیابی‌شده، به سوال کاربر پاسخ بده.\n\n"
        "پاسخ باید:\n"
        "- به فارسی باشد\n"
        "- فقط پاسخ مستقیم سوال را بده، بدون جزئیات تکمیلی یا اضافی\n"
        "- فقط از اسنادی استفاده کن که مستقیماً مرتبط با مقطع یا شرایط ذکر شده در سوال هستند\n"
        "- شماره ماده مرجع را ذکر کند"
    )

    # ------------------------------------------------------------------
    def __init__(
        self,
        llm_api_key: str,
        llm_base_url: str = "https://api.avalai.ir/v1",
        llm_model: str = "qwen3-8b",
        embed_model_name: str = "BAAI/bge-m3",
        max_rounds: int = 3,
        top_k: int = 5,
        qdrant_host: str | None = None,
        qdrant_port: int | None = None,
        embed_model: SentenceTransformer | None = None,
    ):
        self.llm_model = llm_model
        self.max_rounds = max_rounds
        self.top_k = top_k

        self._embed_model = embed_model or SentenceTransformer(embed_model_name)
        self._llm = OpenAI(api_key=llm_api_key, base_url=llm_base_url)
        host = qdrant_host if qdrant_host is not None else QDRANT_HOST
        port = int(qdrant_port) if qdrant_port is not None else QDRANT_PORT
        self._qdrant = QdrantClient(
            url=f"http://{host}:{port}",
            prefer_grpc=False,
            check_compatibility=False,
        )
        # Index must already exist (build it by running: python build_regulation_index.py)
        self._disambiguation = DocDisambiguation(self._llm, self.llm_model)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Used only for final answer generation."""
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

    # ------------------------------------------------------------------
    # Public pipeline methods
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.top_k
        query_vector = self._embed_model.encode(query).tolist()
        response = self._qdrant.query_points(
            collection_name=self.COLLECTION,
            query=query_vector,
            limit=k,
            with_payload=True,
        )
        points = getattr(response, "points", None) or getattr(response, "hits", None) or []
        return [
            {
                "text": (getattr(r, "payload", None) or {}).get("text", ""),
                "source": (getattr(r, "payload", None) or {}).get("source", ""),
                "article_number": (getattr(r, "payload", None) or {}).get("article_number", 0),
                "score": getattr(r, "score", 0.0),
            }
            for r in points
        ]

    def detect_ambiguity(
        self,
        question: str,
        docs: list[dict],
        conversation_history: list[dict] | None = None,
    ) -> dict:
        return self._disambiguation.detect_ambiguity(
            question, docs, conversation_history
        )

    def generate_clarification(
        self,
        question: str,
        docs: list[dict],
        reason: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        return self._disambiguation.generate_clarification(
            question, docs, reason, conversation_history
        )

    def reformulate_question(
        self,
        original_question: str,
        clarification_q: str,
        user_answer: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        return self._disambiguation.reformulate_question(
            original_question, clarification_q, user_answer, conversation_history
        )

    def generate_answer(self, question: str, docs: list[dict]) -> str:
        relevant_docs = filter_docs_by_question(question, docs)
        context = format_documents(relevant_docs)
        prompt = f"سوال: {question}\n\nاسناد مرجع:\n{context}\n\nفقط پاسخ مستقیم سوال را بده."
        return self._call_llm(self.ANSWER_SYSTEM, prompt)

    # ------------------------------------------------------------------
    # High-level: run one disambiguation round
    # ------------------------------------------------------------------

    def run_round(self, question: str, session: SessionState) -> RoundResult:

        session.round_num += 1
        docs = self.retrieve(question)
        docs = filter_docs_by_similarity(docs)  # keep only docs with score >= 40%
        # If no relevant docs, skip doc-based disambiguation and pass question directly to RAG laws
        if no_relevant_docs(docs):
            result = RoundResult(
                question=question,
                docs=docs or [],
                is_ambiguous=False,
                reason="هیچ سند مرتبطی بازیابی نشد؛ سوال بدون بررسی ابهام به مرحله بعد ارسال می‌شود.",
                round_num=session.round_num,
            )
            session.reset()
            return result

        ambiguity = self.detect_ambiguity(question, docs, session.conversation_history)
        is_ambiguous = ambiguity.get("is_ambiguous", False)
        reason = ambiguity.get("reason", "")

        result = RoundResult(
            question=question,
            docs=docs,
            is_ambiguous=is_ambiguous,
            reason=reason,
            round_num=session.round_num,
        )

        if not is_ambiguous or session.round_num > self.max_rounds:
            result.answer = self.generate_answer(question, docs)
            session.reset()
        else:
            clarification_q = self.generate_clarification(
                question, docs, reason, session.conversation_history
            )
            result.clarification_question = clarification_q
            session.phase = "awaiting_clarification"
            session.current_question = question
            session.pending_clarification = clarification_q
            session.pending_docs = docs
            session.pending_reason = reason

        return result

    def handle_clarification_answer(
        self, user_answer: str, session: SessionState
    ) -> tuple[str, RoundResult]:
        session.conversation_history.append(
            {
                "round": session.round_num,
                "question": session.current_question,
                "clarification": session.pending_clarification,
                "user_answer": user_answer,
            }
        )
        new_question = self.reformulate_question(
            session.current_question,
            session.pending_clarification,
            user_answer,
            session.conversation_history,
        )
        result = self.run_round(new_question, session)
        return new_question, result


# ===========================================================================
# Streamlit application
# ===========================================================================

st.set_page_config(
    page_title="سامانه پاسخ‌گویی آیین‌نامه دانشگاه",
    page_icon="📚",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Cached singletons -- load heavy models once to avoid memory bloat and hangs
# ---------------------------------------------------------------------------

EMBED_MODEL_NAME = "BAAI/bge-m3"


@st.cache_resource(show_spinner="در حال بارگذاری مدل امبدینگ (یک بار)...")
def get_embedding_model() -> SentenceTransformer:
    """Single shared embedding model for pipeline, RAG laws, and clarification."""
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner="در حال بارگذاری مدل تشخیص ابهام (یک بار)...")
def get_ambiguity_detector() -> "AmbiguityDetector":
    from ambiguity_detector import AmbiguityDetector
    return AmbiguityDetector("./saved_ambiguity_model")


@st.cache_resource(show_spinner="در حال بارگذاری مدل و اتصال به ایندکس")
def get_pipeline() -> DisambiguationRAG:
    return DisambiguationRAG(
        llm_api_key=os.environ.get('OPENAI_API_KEY'),
        embed_model=get_embedding_model(),
    )


@st.cache_resource(show_spinner="در حال بارگذاری مدل RAG مقررات")
def get_rag_laws() -> RAGLawsHandler:
    return RAGLawsHandler(embedding_model=get_embedding_model())


pipeline = get_pipeline()
rag_laws = get_rag_laws()

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def _create_clarification_handler() -> ClarificationHandler:
    """Use shared cached models to avoid loading 3x embedding + 1x ambiguity model."""
    return ClarificationHandler(
        embedding_model=get_embedding_model(),
        ambiguity_detector=get_ambiguity_detector(),
    )


def reset_conversation():
    st.session_state.messages = []
    st.session_state.session = SessionState()
    if "clarification_handler" in st.session_state:
        st.session_state.clarification_handler.reset_session()
    else:
        st.session_state.clarification_handler = _create_clarification_handler()


# Cap messages to avoid unbounded memory growth
MAX_MESSAGES = 50


def _cap_messages():
    if len(st.session_state.messages) > MAX_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]


if "messages" not in st.session_state:
    st.session_state.session = SessionState()
    st.session_state.messages = []
    st.session_state.clarification_handler = _create_clarification_handler()
else:
    _cap_messages()

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def build_retrieval_details(docs: list[dict]) -> str:
    sources = set(d["source"] for d in docs)
    lines = [f"**تعداد اسناد یافت‌شده:** {len(docs)} سند از {', '.join(sources)}\n"]
    lines.append("| # | منبع | ماده | امتیاز شباهت |")
    lines.append("|---|------|------|-------------|")
    for i, d in enumerate(docs, 1):
        lines.append(
            f"| {i} | آیین‌نامه {d['source']} | ماده {d['article_number']} | {d['score']:.3f} |"
        )
    return "\n".join(lines)


def render_round_result(result: RoundResult, override_answer: str | None = None):
    """Display one round as a clean chatbot message: main reply first, details in an expander."""
    content = override_answer if override_answer else (result.answer if result.answer else result.clarification_question)
    st.session_state.messages.append({"role": "assistant", "content": content})

    # Main reply (chatbot-style)
    st.markdown(content)

    # Optional expander: sources and reasoning
    with st.expander("📎 منابع و جزئیات", expanded=False):
        st.caption(f"سوال این دور: {result.question}")
        sources = ", ".join(d["source"] for d in result.docs[:5])
        st.caption(f"اسناد مرجع: {sources} ({len(result.docs)} سند)")
        if result.is_ambiguous:
            st.caption(f"🔸 وضعیت: مبهم — {result.reason}")
        else:
            st.caption(f"🔹 وضعیت: واضح — {result.reason}")
        for i, d in enumerate(result.docs[:3], 1):
            with st.expander(f"ماده {d['article_number']} — آیین‌نامه {d['source']}", expanded=False):
                st.text(d["text"][:400] + ("..." if len(d["text"]) > 400 else ""))


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

st.markdown(
    '<link href="https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css" rel="stylesheet">',
    unsafe_allow_html=True,
)
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet">',
    unsafe_allow_html=True,
)
st.markdown(
    """
<style>
    html, body, [class*="css"], .stMarkdown, .stMarkdown p, .stMarkdown li,
    .stMarkdown td, .stMarkdown th, .stChatMessage, .stTextInput input,
    button, h1, h2, h3, h4, h5, h6, span, div, label, textarea {
        font-family: 'Vazirmatn', 'Tahoma', "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", "Android Emoji", sans-serif !important;
    }
    [data-testid="stExpander"] summary span:first-child,
    [data-testid="stExpander"] summary [class*="icon"],
    [data-testid="stExpander"] [role="button"] span:first-child,
    [data-testid="stExpander"] [role="button"] [class*="icon"] {
        font-family: "Material Symbols Rounded", "Material Icons", system-ui, sans-serif !important;
    }
    .stChatMessage { direction: rtl; text-align: right; }
    .stElementContainer { direction: rtl; text-align: right; }
    .stChatInput > div { direction: rtl; }
    .block-container { direction: rtl; }
    .stMarkdown table { direction: rtl; text-align: right; width: 100%; }
    .stMarkdown th, .stMarkdown td { text-align: right !important; }
    [data-testid="stStatusWidget"] { direction: rtl; text-align: right; }
    [data-testid="stExpander"] { direction: rtl; text-align: right; }
    pre { direction: ltr; text-align: left; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------

st.markdown(
    '<h2 style="direction: rtl; text-align: right;">سامانه پاسخ‌گویی به سوالات دانشگاهی</h2>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="direction: rtl; text-align: right; color: gray;">سوال خود را درباره دانشگاه بپرسید.</p>',
    unsafe_allow_html=True,
)

if st.button("شروع مکالمه جدید"):
    reset_conversation()
    st.rerun()

for msg in st.session_state.messages:
    role = msg["role"]
    avatar = "👤" if role == "user" else "📚"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input handling
# Pipeline: user → clarification_module (until clear) → app doc-based disambiguation → RAG laws → answer
# ---------------------------------------------------------------------------

user_input = st.chat_input("سوال خود را بنویسید...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    sess = st.session_state.session
    our_clarification = st.session_state.clarification_handler

    with st.chat_message("assistant", avatar="📚"):
        loading_ph = st.empty()
        loading_ph.markdown("⏳ در حال پردازش...")

        # --- Layer 2: App's doc-based clarification (uses retrieved docs to disambiguate) ---
        if sess.phase == "awaiting_clarification":
            new_question, result = pipeline.handle_clarification_answer(
                user_input, sess
            )
            loading_ph.empty()
            st.markdown(f"سوال بازنویسی‌شده: **{new_question}**")
            if result.answer is not None:
                final_answer, _ = rag_laws.answer(result.question)
                render_round_result(result, override_answer=final_answer)
                sess.reset()
                our_clarification.reset_session()
            elif not result.is_ambiguous:
                final_answer, _ = rag_laws.answer(result.question)
                render_round_result(result, override_answer=final_answer)
                sess.reset()
                our_clarification.reset_session()
            else:
                render_round_result(result)
            st.stop()

        response, clarification_done = our_clarification.process(user_input)
        if not clarification_done:
            loading_ph.empty()
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.caption("🔍 روشن‌سازی سوال (مرحله ۱)")
            st.markdown(response)
            st.stop()
        # Clarification_module done → clarified question
        clarified_question = response
        our_clarification.reset_session()
        loading_ph.empty()
        st.markdown(f"**سوال روشن‌شده:** {clarified_question}")
        st.caption("مرحله ۲: بررسی ابهام بر اساس اسناد بازیابی‌شده")

        # --- Run app's doc-based disambiguation round with clarified question ---
        result = pipeline.run_round(clarified_question, sess)

        if result.answer is not None:
            # Final answer from RAG laws (not from pipeline's own generate_answer)
            final_answer, _ = rag_laws.answer(result.question)
            render_round_result(result, override_answer=final_answer)
            sess.reset()
            our_clarification.reset_session()
        elif not result.is_ambiguous:
            # No relevant docs: skip doc-based clarification, pass question directly to RAG laws
            final_answer, _ = rag_laws.answer(result.question)
            render_round_result(result, override_answer=final_answer)
            sess.reset()
            our_clarification.reset_session()
        else:
            render_round_result(result)
