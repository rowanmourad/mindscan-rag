"""
retrieval.py
------------
PURPOSE:
This file performs semantic retrieval from the Chroma vector database.

Pipeline:

User Query
      ↓
Convert Query into Embedding
      ↓
Semantic Similarity Search
      ↓
Filter by Tumor Category
      ↓
Retrieve Top-k Most Relevant Chunks
      ↓
Send Retrieved Context to the LLM

Instead of searching for exact keywords, the system searches for
documents with the closest semantic meaning to the user's query.
"""

import pandas as pd


# ==========================================================
# STEP 1 — SEMANTIC RETRIEVAL
# ==========================================================
# Purpose:
# Finds the most relevant medical knowledge from the vector
# database based on the user's question.
#
# Why?
#
# Large Language Models should not answer medical questions
# using only their pretrained knowledge.
#
# Instead, they first retrieve evidence from trusted medical
# research papers, then generate the final response based on
# that retrieved context.
def retrieve_knowledge(collection, query: str, predicted_class: str = None,
                        k: int = 5, include_general: bool = True) -> pd.DataFrame:         #How many results should be returned? 5, 
    """
    Retrieves the top-k most relevant chunks for `query`.

    If predicted_class is given, results are restricted to that category
    plus (optionally) the general_overview / mri_diagnosis categories.
    """
   # ======================================================
    # STEP 2 — BUILD CATEGORY FILTER
    # ======================================================
    # Purpose:
    # Restrict the search to only relevant medical papers.
    #
    # Why?
    #
    # Suppose the MRI classifier predicts "Glioma".
    #
    # Instead of searching every paper in the knowledge base,
    # we search only:
    #
    # • Glioma papers
    # • General brain tumor papers
    # • MRI diagnosis papers
    #
    # This improves retrieval accuracy and avoids returning
    # information about unrelated tumors.
    where_filter = None

    if predicted_class:
        categories = [predicted_class]
        if include_general:
            categories += ["general_overview", "mri_diagnosis"]
        where_filter = {"category": {"$in": categories}}
      
# ======================================================
    # STEP 3 — SEMANTIC SEARCH
    # ======================================================
    # Purpose:
    # Search the vector database using embeddings instead of
    # exact words.
    #
    # Why?
    #
    # Two sentences may have different wording but identical
    # meaning.
    #
    # Embedding search retrieves semantically similar chunks,
    # making retrieval much more accurate than keyword search.
                          
    results = collection.query(
        query_texts=[query],
        n_results=k,                          # Number of retrieved chunks.
        where=where_filter,                    # Apply category filter.
    )

    # ======================================================
    # STEP 4 — ORGANIZE RESULTS
    # ======================================================
    # Purpose:
    # Convert ChromaDB's raw output into a structured table.
    #
    # Why?
    #
    # Chroma returns nested dictionaries and lists.
    #
    # Converting them into a DataFrame makes the retrieved
    # knowledge easier to inspect, debug, and pass to the
    # next stage of the Graph RAG pipeline.  
                          
    rows = []
    if results["ids"] and results["ids"][0]:              #This loops through every retrieved chunk,to give me the top 5 most relevant chunks.
        for i in range(len(results["ids"][0])):
            rows.append({
                "chunk_id": results["ids"][0][i],
                "category": results["metadatas"][0][i]["category"],
                "source_file": results["metadatas"][0][i]["source_file"],
                "page_number": results["metadatas"][0][i]["page_number"],
                "distance": results["distances"][0][i],
                "text": results["documents"][0][i],
            })

    return pd.DataFrame(rows)
