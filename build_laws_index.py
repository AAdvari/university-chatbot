"""
Build two Qdrant indices from laws DOCX files (bachelor and master).
Uses LLM (qwen3-32b) to chunk documents properly and produce concise embedding summaries;
embeds the summaries and stores full chunk text in payload for retrieval.
"""
import json
import os
import re
import time
import urllib.request
import dotenv
dotenv.load_dotenv()
from pathlib import Path

from docx import Document
from openai import OpenAI
from qdrant_client import QdrantClient

from llm_utils import chat_completion_with_thinking
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer


LAWS_DIR = Path(__file__).resolve().parent / "laws"
EMBEDDING_MODEL = "BAAI/bge-m3"
QDRANT_HOST = os.environ.get("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
LAWS_LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "qwen3-32b")
COLLECTION_BACHELOR = "bachelor_laws"
COLLECTION_MASTER = "master_laws"

# Max characters per LLM call to avoid context overflow
MAX_TEXT_PER_CALL = 14_000


def _qdrant_http_ok(host: str, port: int) -> bool:
    try:
        req = urllib.request.Request(f"http://{host}:{port}/", method="GET")
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status in (200, 404)
    except Exception:
        return False


def load_docx_text(path: Path) -> str:
    """Extract plain text from a DOCX file."""
    doc = Document(path)
    return "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())


def chunk_and_summarize_with_llm(
    client: OpenAI, model: str, full_text: str, level: str
) -> list[dict]:
    """
    Ask LLM to (1) chunk the document properly (by article / logical segment),
    and (2) for each chunk give a concise representation (summary) for embedding.
    Returns list of {chunk_id, full_text, embedding_summary}.
    """
    if not full_text.strip():
        return []
    text_slice = full_text[:MAX_TEXT_PER_CALL]
    prompt = f"""متن زیر بخشی از آیین‌نامه یا قوانین ({level}) است.

وظیفه:
۱) متن را به قطعات معنادار تقسیم کن (بر اساس ماده‌های قانونی یا بخش‌های منطقی). اگر یک ماده خیلی طولانی است آن را به زیربخش‌های منطقی تقسیم کن.
۲) برای هر قطعه دو خروجی بده:
   - full_text: عین متن آن قطعه (بدون کم و زیاد).
   - embedding_summary: یک خلاصه یا بیان بسیار مختصر (یک تا حداکثر سه جمله) که موضوع و محتوای آن قطعه را برای جستجوی معنایی مناسب کند؛ مثلاً «شرایط ترمیم واحد در مقطع کارشناسی» یا «حداکثر تعداد واحد قابل اخذ در ترم».

خروجی فقط یک آرایه JSON باشد، هر عنصر به این شکل:
{{"chunk_id": "ماده_۱ یا ماده_۲_بخش_۱", "full_text": "متن کامل قطعه", "embedding_summary": "خلاصه برای جستجو"}}
فقط آرایه JSON، بدون توضیح یا markdown.

متن:
{text_slice}"""
    try:
        content, _ = chat_completion_with_thinking(
            client,
            model,
            [
                {"role": "system", "content": "You output only a valid JSON array. No markdown, no explanation, no code fence."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        content = (content or "").strip()
        content = re.sub(r"^```\w*\n?", "", content).replace("```", "").strip()
        items = json.loads(content)
        if isinstance(items, list) and items:
            out = []
            for i, x in enumerate(items, 1):
                chunk_id = str(x.get("chunk_id", f"chunk_{i}"))
                full = str(x.get("full_text", "")).strip()
                summary = str(x.get("embedding_summary", "")).strip()
                if not full and not summary:
                    continue
                if not summary:
                    summary = full[:500]
                out.append({
                    "chunk_id": chunk_id,
                    "full_text": full,
                    "embedding_summary": summary,
                })
            return out
    except Exception:
        pass
    return _chunk_and_summarize_fallback(client, model, full_text, level)


def _chunk_and_summarize_fallback(
    client: OpenAI, model: str, full_text: str, level: str
) -> list[dict]:
    """Fallback: split by ماده N, then ask LLM for embedding_summary per chunk."""
    article_pattern = re.compile(r"\s*ماده\s*[۰-۹\d]+\s*[:\-]?", re.IGNORECASE)
    parts = article_pattern.split(full_text)
    chunks = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part or len(part) < 20:
            continue
        chunk_id = f"matter_{i}"
        summary = _llm_summarize_chunk(client, model, part, chunk_id)
        chunks.append({
            "chunk_id": chunk_id,
            "full_text": part,
            "embedding_summary": summary or part[:500],
        })
    if not chunks:
        chunks = [{"chunk_id": "doc_1", "full_text": full_text, "embedding_summary": full_text[:500]}]
    return chunks


def _llm_summarize_chunk(client: OpenAI, model: str, text: str, chunk_id: str) -> str:
    """Get a short embedding_summary for one chunk."""
    if len(text) > 2500:
        text = text[:2500] + "..."
    try:
        out, _ = chat_completion_with_thinking(
            client,
            model,
            [
                {"role": "system", "content": "فقط یک یا دو جمله خلاصه بنویس. بدون عنوان یا توضیح اضافه."},
                {"role": "user", "content": f"""این قطعه از آیین‌نامه ({chunk_id}) را در یک یا دو جمله خلاصه کن به گونه‌ای که برای جستجوی معنایی مناسب باشد:\n\n{text}"""},
            ],
            temperature=0.2,
        )
        out = (out or "").strip()
        if out:
            return out[:800]
    except Exception:
        pass
    return ""


def main():
    bachelor_path = LAWS_DIR / "bachelor.docx"
    master_path = LAWS_DIR / "master.docx"
    if not bachelor_path.exists() and not master_path.exists():
        raise SystemExit(
            f"Place bachelor.docx and/or master.docx in {LAWS_DIR}. See {LAWS_DIR / 'README.md'}."
        )

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.avalai.ir/v1"),
    )

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    dim = model.get_sentence_embedding_dimension()

    qdrant_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
    deadline = time.monotonic() + 60
    while not _qdrant_http_ok(QDRANT_HOST, QDRANT_PORT):
        if time.monotonic() > deadline:
            raise SystemExit("Qdrant not reachable.")
        print("  Waiting for Qdrant...")
        time.sleep(2)
    qdrant = QdrantClient(url=qdrant_url, prefer_grpc=False, check_compatibility=False)

    for collection_name, docx_path, level in [
        (COLLECTION_BACHELOR, bachelor_path, "کارشناسی"),
        (COLLECTION_MASTER, master_path, "کارشناسی ارشد"),
    ]:
        if not docx_path.exists():
            print(f"Skipping {docx_path} (not found).")
            continue
        print(f"Processing {docx_path}...")
        full_text = load_docx_text(docx_path)
        if not full_text.strip():
            print(f"  No text in {docx_path}. Skipping.")
            continue
        print("  Chunking and getting embedding summaries (LLM)...")
        chunks = chunk_and_summarize_with_llm(client, LAWS_LLM_MODEL, full_text, level)
        if not chunks:
            print(f"  No chunks from LLM. Skipping.")
            continue
        print(f"  Got {len(chunks)} chunks. Embedding summaries...")
        texts_to_embed = [c["embedding_summary"] for c in chunks]
        payloads = [
            {
                "chunk_id": c["chunk_id"],
                "article_id": c["chunk_id"],
                "text": c["full_text"],
                "embedding_summary": c["embedding_summary"],
                "level": level,
                "source": docx_path.name,
            }
            for c in chunks
        ]
        print("  Embedding...")
        vectors = model.encode(texts_to_embed, show_progress_bar=True).tolist()
        if qdrant.collection_exists(collection_name):
            qdrant.delete_collection(collection_name)
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=i, vector=vectors[i], payload=payloads[i])
            for i in range(len(vectors))
        ]
        qdrant.upsert(collection_name=collection_name, points=points)
        print(f"  Indexed {len(points)} points in {collection_name}.")

    print("Done.")


if __name__ == "__main__":
    main()
