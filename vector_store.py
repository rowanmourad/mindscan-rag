"""
vector_store.py
----------------
PURPOSE:
This file creates and manages the Vector Database used by Graph RAG.

Pipeline:

Text Chunks
      ↓
Generate Embeddings
      ↓
Store in ChromaDB
      ↓
Semantic Search
      ↓
Retrieve Most Relevant Chunks

Instead of searching documents by exact keywords, the vector database
stores numerical representations (embeddings) of every chunk so that
similar meanings are located even if different words are used.
"""

import chromadb                             # Vector database used for semantic retrieval.
import pandas as pd                         # Handles chunk data as DataFrames.
from chromadb.utils import embedding_functions

# Configuration settings:
# CHROMA_DIR -> where the database is stored.
# EMBEDDING_MODEL_NAME -> biomedical embedding model. 
from config import CHROMA_DIR, EMBEDDING_MODEL_NAME

# ==========================================================
# COLLECTION NAME
# ==========================================================
# Purpose:
# Gives the knowledge base a fixed name inside ChromaDB.
#
# Every chunk from all medical papers will be stored inside
# this collection.
COLLECTION_NAME = "brain_tumor_kb"


# ==========================================================
# STEP 1 — CREATE OR LOAD THE VECTOR DATABASE
# ==========================================================
# Purpose:
# Opens an existing Chroma database if it already exists.
#
# If this is the first run, a new collection is created.
#
# It also loads the embedding model that converts every text
# chunk into a numerical vector.
#
# The similarity metric is Cosine Similarity, which measures
# how close two embedding vectors are in semantic meaning.

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

# ==========================================================
# STEP 2 — ADD NEW CHUNKS
# ==========================================================
# Purpose:
# Adds only NEW chunks into the vector database.
#
# Why?
#
# Every time the program runs, we don't want to store the same
# chunk twice.
#
# The function first checks which chunk IDs already exist,
# then inserts only the missing ones.
#
# This property is called "Idempotency", meaning the function
# can be executed repeatedly without creating duplicates.

def add_new_chunks(collection, chunks_df: pd.DataFrame, batch_size: int = 100) -> None:
    """
    Embeds and adds only the chunks that aren't already stored (idempotent --
    safe to re-run on the same PDF library without creating duplicates).
    """
    # Verify that all required information exists.
    
    required_cols = {"chunk_id", "chunk_text", "category", "source_file", "page_number"}
    missing_cols = required_cols - set(chunks_df.columns)

    if len(chunks_df) == 0 or missing_cols:
        print("chunks_df is empty or missing expected columns -- nothing to add yet.")
        print("chunks_df shape:", chunks_df.shape)
        print("This usually means no PDF pages were found -- check your papers/<category>/ folders.")
        return

    # ------------------------------------------------------
    # Retrieve all existing chunk IDs.
    # ------------------------------------------------------
    existing_ids = set()
    if collection.count() > 0:
        existing_ids = set(collection.get(include=[])["ids"])

    new_chunks_df = chunks_df[~chunks_df["chunk_id"].isin(existing_ids)]

    # ------------------------------------------------------
    # Insert chunks in small batches.
    # ------------------------------------------------------
    # Why batches?
    #
    # Thousands of embeddings at once consume a lot of memory.
    #
    # Processing batches improves efficiency and stability.
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

# ==========================================================
# STEP 3 — CLEAN METADATA
# ==========================================================
# Purpose:
# Sometimes category names may accidentally contain extra
# spaces, for example:
#
# "glioma "
#
# instead of
#
# "glioma"
#
# Even a small space causes filtering by category to fail.
#
# This function cleans every stored category label by removing
# unnecessary whitespace and updating the database.

def normalize_category_metadata(collection) -> None:
    """One-off cleanup: strips stray whitespace from stored category labels."""
    all_data = collection.get(include=["metadatas"])
    print("Total records:", len(all_data["ids"]))

    for id_, meta in zip(all_data["ids"], all_data["metadatas"]):
        meta["category"] = meta["category"].strip()
        collection.update(ids=[id_], metadatas=[meta])

    sample = collection.get(limit=20, include=["metadatas"])
    print("Categories present:", set(m["category"] for m in sample["metadatas"]))
