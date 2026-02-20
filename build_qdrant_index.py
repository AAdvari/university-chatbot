"""
Build Qdrant vector index from ambiguous questions (is_ambiguous=1) in train.json and test.json.
Uses 'question' for embedding and stores 'clarifications' as metadata.
"""
import json
import os
import time
import urllib.request
import urllib.error
import dotenv
dotenv.load_dotenv()
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "ambiguous_questions"
EMBEDDING_MODEL = "BAAI/bge-m3"  # 768d, native sentence-transformers, Persian-friendly
# EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"  # 768d, native sentence-transformers, Persian-friendly
QDRANT_HOST = os.environ.get("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_RETRY_SEC = int(os.environ.get("QDRANT_RETRY_SEC", "60"))
DATA_DIR = Path(__file__).resolve().parent / "data"


def _qdrant_http_ok(host: str, port: int) -> bool:
    """Check if Qdrant REST API responds (avoid 502 from proxy/client)."""
    try:
        req = urllib.request.Request(
            f"http://{host}:{port}/",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status in (200, 404)  # Qdrant root may 404, still means server is up
    except Exception:
        return False


def load_ambiguous_questions():
    """Load all items with is_ambiguous=1 from train.json and test.json."""
    records = []
    for filename in ("train.json", "test.json"):
        path = DATA_DIR / filename
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if item.get("is_ambiguous") != 1:
                continue
            question = item.get("question", "").strip()
            clarifications = item.get("clarifications") or []
            if not question:
                continue
            records.append({
                "question": question,
                "clarifications": clarifications,
                "id": item.get("id", str(len(records))),
            })
    return records


def main():
    print("Loading ambiguous questions...")
    records = load_ambiguous_questions()
    if not records:
        raise SystemExit("No ambiguous questions found in data/train.json or data/test.json.")

    print(f"Loaded {len(records)} ambiguous questions. Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    dim = model.get_sentence_embedding_dimension()

    print("Computing embeddings...")
    questions = [r["question"] for r in records]
    vectors = model.encode(questions, show_progress_bar=True).tolist()

    qdrant_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
    print("Connecting to Qdrant...")
    deadline = time.monotonic() + QDRANT_RETRY_SEC
    while True:
        if _qdrant_http_ok(QDRANT_HOST, QDRANT_PORT):
            break
        if time.monotonic() > deadline:
            print(
                "\nQdrant did not become reachable. Please check:\n"
                "  1. Start Qdrant:  docker compose up -d\n"
                "  2. Verify:        curl -s " + qdrant_url + "\n"
                "  3. If using a remote host, set QDRANT_HOST and QDRANT_PORT."
            )
            raise SystemExit(1)
        print(f"  Waiting for Qdrant at {qdrant_url}...")
        time.sleep(2)

    client = QdrantClient(
        url=qdrant_url,
        prefer_grpc=False,
        check_compatibility=False,
    )

    if client.collection_exists(COLLECTION_NAME):
        print(f"Recreating collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=i,
            vector=vec,
            payload={
                "question": rec["question"],
                "clarifications": rec["clarifications"],
                "source_id": rec["id"],
            },
        )
        for i, (rec, vec) in enumerate(zip(records, vectors))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Indexed {len(points)} points into collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
