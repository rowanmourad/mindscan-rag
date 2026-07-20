"""
retrieval.py
------------
Semantic search over the Chroma knowledge base, filtered by tumor category.
"""

import pandas as pd


def retrieve_knowledge(collection, query: str, predicted_class: str = None,
                        k: int = 5, include_general: bool = True) -> pd.DataFrame:
    """
    Retrieves the top-k most relevant chunks for `query`.

    If predicted_class is given, results are restricted to that category
    plus (optionally) the general_overview / mri_diagnosis categories.
    """
    where_filter = None

    if predicted_class:
        categories = [predicted_class]
        if include_general:
            categories += ["general_overview", "mri_diagnosis"]
        where_filter = {"category": {"$in": categories}}

    results = collection.query(
        query_texts=[query],
        n_results=k,
        where=where_filter,
    )

    rows = []
    if results["ids"] and results["ids"][0]:
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
