"""
main.py
-------
Command-line entry point for the MindScan Brain-Tumor RAG assistant.

Typical usage
=============

1) Build/update the knowledge base from PDFs in data/papers/<category>/:

    python main.py build-kb

2) Generate a full AI-assisted report for an MRI image:

    python main.py report --image data/sample_images/glioma_sample1.jpg \\
        --context "45-year-old patient, headache and visual disturbance for 3 weeks."

3) Ask a free-text follow-up question (optionally tied to a prior prediction):

    python main.py ask --question "Why is the confidence only this high?"

Configuration
=============
All paths are read from config.py and can be overridden with environment
variables (MINDSCAN_BASE_DIR, MINDSCAN_REPO_DIR, etc.) -- see config.py for
details. Set your OpenRouter key with:

    export OPENROUTER_API_KEY="sk-or-..."

(or you'll be prompted for it interactively).
"""

import argparse

import config
from ingest import build_chunks_dataframe, load_pdfs_from_folder
from llm_client import test_openrouter_connection
from model_pipeline import predict_tumor
from qa import ask_question
from report_generator import generate_medical_report
from vector_store import add_new_chunks, get_collection, normalize_category_metadata


def cmd_build_kb(args):
    """Loads PDFs, chunks them, and (re-)embeds them into the vector DB."""
    config.ensure_directories()

    pdf_records = load_pdfs_from_folder(config.PDF_DIR)
    print(f"Loaded {len(pdf_records)} usable pages.")
    if not pdf_records:
        print(f"Add PDFs under {config.PDF_DIR}/<category>/ and re-run this command.")
        return

    chunks_df = build_chunks_dataframe(pdf_records)
    print(f"Total chunks: {len(chunks_df)}")

    collection = get_collection()
    add_new_chunks(collection, chunks_df)
    normalize_category_metadata(collection)


def cmd_test_llm(args):
    """Confirms the OpenRouter API key + model are working."""
    api_key = config.get_openrouter_api_key()
    test_openrouter_connection(api_key)


def cmd_report(args):
    """Runs the full pipeline: classify -> retrieve -> draft report -> PDF."""
    config.ensure_directories()
    api_key = config.get_openrouter_api_key()
    collection = get_collection()

    result = generate_medical_report(
        collection=collection,
        api_key=api_key,
        image_path=args.image,
        patient_context=args.context or "",
        top_k=args.top_k,
    )

    print("\n=== Prediction ===")
    print(result["prediction"]["predicted_class"], "-",
          f"{result['prediction']['confidence'] * 100:.1f}% confidence")

    print("\n=== Retrieved Sources ===")
    print(result["retrieved_chunks"][["category", "source_file", "page_number", "distance"]])

    print("\n=== Report (structured) ===")
    for key, value in result["report_data"].items():
        print(f"-- {key.upper()} --\n{value}\n")

    if result["pdf_path"]:
        print("PDF saved to:", result["pdf_path"])


def cmd_ask(args):
    """Asks a free-text follow-up question, optionally scoped to a tumor category."""
    config.ensure_directories()
    api_key = config.get_openrouter_api_key()
    collection = get_collection()

    prediction = None
    if args.image:
        prediction = predict_tumor(args.image)

    result = ask_question(
        collection=collection,
        api_key=api_key,
        question=args.question,
        prediction=prediction,
        top_k=args.top_k,
    )

    print("\n=== Question ===")
    print(result["question"])
    print("\n=== Retrieved Sources ===")
    print(result["retrieved_chunks"][["category", "source_file", "page_number", "distance"]])
    print("\n=== Answer ===")
    print(result["answer"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MindScan Brain-Tumor RAG decision-support assistant."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_build = subparsers.add_parser("build-kb", help="Build/update the PDF knowledge base.")
    p_build.set_defaults(func=cmd_build_kb)

    p_test = subparsers.add_parser("test-llm", help="Test the OpenRouter connection.")
    p_test.set_defaults(func=cmd_test_llm)

    p_report = subparsers.add_parser("report", help="Generate a full AI-assisted report for an MRI image.")
    p_report.add_argument("--image", required=True, help="Path to the MRI image.")
    p_report.add_argument("--context", default="", help="Free-text patient context.")
    p_report.add_argument("--top-k", type=int, default=5, dest="top_k")
    p_report.set_defaults(func=cmd_report)

    p_ask = subparsers.add_parser("ask", help="Ask a free-text follow-up question.")
    p_ask.add_argument("--question", required=True, help="The doctor's question.")
    p_ask.add_argument("--image", default=None,
                        help="Optional MRI image to classify first, so the answer "
                             "is grounded in this patient's prediction.")
    p_ask.add_argument("--top-k", type=int, default=5, dest="top_k")
    p_ask.set_defaults(func=cmd_ask)

    return parser


if __name__ == "__main__":
    arg_parser = build_arg_parser()
    parsed_args = arg_parser.parse_args()
    parsed_args.func(parsed_args)
