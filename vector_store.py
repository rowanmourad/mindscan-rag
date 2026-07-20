"""
vector_store.py
----------------
Builds/loads the persistent Chroma vector database and embeds new PDF
chunks into it using a biomedical sentence-transformer model.
"""

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions

from config import CHROMA_DIR, EMBEDDING_MODEL_NAME

COLLECTION_NAME = "brain_tumor_kb"


def get_collection(chroma_dir=CHROMA_DIR, embedding_model_name: str = EMBEDDING_MODEL_NAME):
    """Opens (or creates) the persistent Chroma collection used for retrieval."""
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model_name
    )
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    print("Chunks already stored in persistent DB:", collection.count())
    return collection


def add_new_chunks(collection, chunks_df: pd.DataFrame, batch_size: int = 100) -> None:
    """
    Embeds and adds only the chunks that aren't already stored (idempotent --
    safe to re-run on the same PDF library without creating duplicates).
    """
    required_cols = {"chunk_id", "chunk_text", "category", "source_file", "page_number"}
    missing_cols = required_cols - set(chunks_df.columns)

    if len(chunks_df) == 0 or missing_cols:
        print("chunks_df is empty or missing expected columns -- nothing to add yet.")
        print("chunks_df shape:", chunks_df.shape)
        print("This usually means no PDF pages were found -- check your papers/<category>/ folders.")
        return

    existing_ids = set()
    if collection.count() > 0:
        existing_ids = set(collection.get(include=[])["ids"])

    new_chunks_df = chunks_df[~chunks_df["chunk_id"].isin(existing_ids)]

    if len(new_chunks_df) > 0:
        for start in range(0, len(new_chunks_df), batch_size):
            batch = new_chunks_df.iloc[start:start + batch_size]
            collection.add(
                ids=batch["chunk_id"].tolist(),
                documents=batch["chunk_text"].tolist(),
                metadatas=batch[["category", "source_file", "page_number"]].to_dict("records"),
            )
        print(f"Added {len(new_chunks_df)} new chunks.")
    else:
        print("No new chunks to add.")

    print("Total chunks now in persistent collection:", collection.count())


def normalize_category_metadata(collection) -> None:
    """One-off cleanup: strips stray whitespace from stored category labels."""
    all_data = collection.get(include=["metadatas"])
    print("Total records:", len(all_data["ids"]))

    for id_, meta in zip(all_data["ids"], all_data["metadatas"]):
        meta["category"] = meta["category"].strip()
        collection.update(ids=[id_], metadatas=[meta])

    sample = collection.get(limit=20, include=["metadatas"])
    print("Categories present:", set(m["category"] for m in sample["metadatas"]))
