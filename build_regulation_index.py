"""
Document loading, chunking by article, and Qdrant indexing for university regulations.
Used by app.py to build the university_regulations collection.
"""
import re
from dataclasses import dataclass

import docx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


@dataclass
class Chunk:
    text: str
    source: str
    article_number: int
    chunk_id: int = 0


def load_docx(filepath: str) -> list[str]:
    """Load a DOCX file and return list of non-empty paragraph texts."""
    doc = docx.Document(filepath)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]


def chunk_by_article(paragraphs: list[str], source: str) -> list[Chunk]:
    """Split paragraphs into chunks by law article (ماده N)."""
    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_article = 0
    article_pattern = re.compile(r"ماده\s+[\u06F0-\u06F9۰-۹0-9]+")

    for para in paragraphs:
        match = article_pattern.search(para)
        if match and "عبارت است از" in para:
            if current_lines and current_article > 0:
                chunks.append(
                    Chunk(
                        text="\n".join(current_lines),
                        source=source,
                        article_number=current_article,
                    )
                )
            current_lines = [para]
            num_str = re.findall(r"[\u06F0-\u06F9۰-۹0-9]+", para)
            if num_str:
                current_article = int(
                    num_str[0].translate(
                        str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
                    )
                )
        else:
            current_lines.append(para)

    if current_lines and current_article > 0:
        chunks.append(
            Chunk(
                text="\n".join(current_lines),
                source=source,
                article_number=current_article,
            )
        )
    return chunks


def index_documents(
    doc_paths: list[tuple[str, str]],
    embed_model,
    qdrant: QdrantClient,
    collection_name: str = "university_regulations",
) -> None:
    """
    Load DOCX files, chunk by article, embed with embed_model, and upsert into Qdrant.
    Recreates the collection if it already exists.
    """
    all_chunks: list[Chunk] = []
    for path, source_label in doc_paths:
        paragraphs = load_docx(path)
        all_chunks.extend(chunk_by_article(paragraphs, source=source_label))
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_id = i

    dim = embed_model.get_sentence_embedding_dimension()
    if qdrant.collection_exists(collection_name):
        qdrant.delete_collection(collection_name)
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    texts = [c.text for c in all_chunks]
    embeddings = embed_model.encode(texts, batch_size=16)
    points = [
        PointStruct(
            id=c.chunk_id,
            vector=emb.tolist(),
            payload={
                "text": c.text,
                "source": c.source,
                "article_number": c.article_number,
            },
        )
        for c, emb in zip(all_chunks, embeddings)
    ]
    qdrant.upsert(collection_name=collection_name, points=points)


# ---------------------------------------------------------------------------
# Standalone: run indexing before starting the app
# ---------------------------------------------------------------------------

DEFAULT_DOC_PATHS = [
    ("./laws/master.docx", "کارشناسی ارشد"),
    ("./laws/bachelor.docx", "کارشناسی"),
]
COLLECTION_NAME = "university_regulations"
EMBED_MODEL_NAME = "BAAI/bge-m3"


def main():
    import os
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient

    try:
        import dotenv
        dotenv.load_dotenv()
    except ImportError:
        pass

    host = os.environ.get("QDRANT_HOST", "127.0.0.1")
    port = int(os.environ.get("QDRANT_PORT", "6333"))

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    print("Connecting to Qdrant...")
    qdrant = QdrantClient(
        url=f"http://{host}:{port}",
        prefer_grpc=False,
        check_compatibility=False,
    )
    print("Indexing university regulations...")
    index_documents(DEFAULT_DOC_PATHS, embed_model, qdrant, COLLECTION_NAME)
    print("Done. You can now run app.py.")


if __name__ == "__main__":
    main()
