"""
app.py
------
Streamlit interface for the MindScan Brain-Tumor RAG assistant.
 
Run with:
    streamlit run app.py
 
Reads OPENROUTER_API_KEY from the environment (see .env.example). If it's
not set, you'll be prompted to enter it in the sidebar instead.
"""
 
# --------------------------------------------------------------------------
# Environment setup -- MUST run before any TensorFlow/PyTorch/OpenCV import
# (including transitively, via streamlit or other modules below). These
# libraries read these env vars at import time to configure their native
# (C/C++) runtimes; setting them later has no effect.
#
#   OMP_NUM_THREADS=1        - avoids thread-pool contention between
#                               TensorFlow's and PyTorch's own OpenMP pools
#   KMP_DUPLICATE_LIB_OK=1   - works around a duplicate OpenMP runtime
#                               (libiomp/libgomp) being loaded by both
#                               tensorflow and torch in the same process,
#                               a common cause of native segfaults
#   CUDA_VISIBLE_DEVICES=-1  - forces CPU-only execution (no GPU on this
#                               deployment; avoids CUDA init attempts/errors)
#   TF_CPP_MIN_LOG_LEVEL=3   - quiets TensorFlow's C++ log spam
# --------------------------------------------------------------------------
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
 
# PyTorch must be imported before Keras/TensorFlow in this process. When both
# ML frameworks load their native (C/C++) runtimes in the same process, doing
# it in the wrong order is a known cause of hard segfaults at import time
# (before any app code even runs). Importing torch first here, ahead of
# model_pipeline (which pulls in keras -> tensorflow) and vector_store (which
# pulls in chromadb -> sentence-transformers -> torch), keeps the load order
# stable.
import torch  # noqa: F401
 
import tempfile
 
import streamlit as st
 
import config
from model_pipeline import predict_tumor
from qa import ask_question
from report_generator import generate_medical_report
from vector_store import get_collection
 
st.set_page_config(page_title="MindScan", page_icon="🧠", layout="wide")
 
config.ensure_directories()
 
 
# --------------------------------------------------------------------------
# Cached resources -- loaded once per server process, not on every rerun.
# --------------------------------------------------------------------------
 
@st.cache_resource(show_spinner="Loading knowledge base...")
def _load_collection():
    return get_collection()
 
 
@st.cache_resource(show_spinner="Loading tumor-classification models...")
def _warm_up_model():
    # Importing here (not at module load) keeps app startup fast until the
    # first prediction is actually needed, then caches the loaded pipeline.
    from model_pipeline import load_pipeline
    return load_pipeline()
 
 
# --------------------------------------------------------------------------
# Sidebar -- API key + settings
# --------------------------------------------------------------------------
 
st.sidebar.title("Settings")
 
api_key = st.secrets.get("OPENROUTER_API_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input(
        "OpenRouter API key",
        type="password",
        help="Get a free key at https://openrouter.ai/keys. Also enable "
             "'Free endpoints that may train on request data' at "
             "https://openrouter.ai/settings/privacy to use :free models.",
    )
 
top_k = st.sidebar.slider("Number of literature sources to retrieve", 1, 10, 5)
 
kb_size_placeholder = st.sidebar.empty()
 
st.sidebar.markdown("---")
st.sidebar.caption(
    "Research/educational tool -- not a medical diagnosis. "
    "All findings require clinician confirmation."
)
 
 
# --------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------
 
st.title("🧠 MindScan: Brain Tumor Classification + Clinical Report Assistant")
st.caption("AI-assisted MRI classification with literature-grounded draft reports.")
 
if not api_key:
    st.warning("Enter your OpenRouter API key in the sidebar to generate reports or ask questions.")
 
tab_report, tab_ask = st.tabs(["📄 Generate Report", "💬 Ask a Follow-up Question"])
 
 
# --------------------------------------------------------------------------
# Tab 1: Generate a full report
# --------------------------------------------------------------------------
 
with tab_report:
    col_input, col_output = st.columns([1, 2])
 
    with col_input:
        uploaded_image = st.file_uploader(
            "Upload MRI image", type=["jpg", "jpeg", "png"], key="report_image"
        )
        patient_context = st.text_area(
            "Patient context (optional)",
            placeholder="e.g. 45-year-old patient, headache and visual disturbance for 3 weeks.",
            key="report_context",
        )
        generate_clicked = st.button("Generate Report", type="primary", use_container_width=True)
 
    with col_output:
        if generate_clicked:
            if not api_key:
                st.error("Please enter your OpenRouter API key in the sidebar first.")
            elif uploaded_image is None:
                st.error("Please upload an MRI image.")
            else:
                # predict_tumor()/generate_medical_report() need a real file path.
                suffix = os.path.splitext(uploaded_image.name)[1] or ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_image.getbuffer())
                    tmp_path = tmp.name
 
                collection = _load_collection()
                kb_size_placeholder.caption(f"Knowledge base: {collection.count()} chunks")
 
                with st.spinner("Classifying image, retrieving literature, and drafting report..."):
                    try:
                        result = generate_medical_report(
                            collection=collection,
                            api_key=api_key,
                            image_path=tmp_path,
                            patient_context=patient_context or "",
                            top_k=top_k,
                        )
                        st.session_state["last_result"] = result
                    except Exception as e:
                        st.error(f"Report generation failed: {e}")
                        result = None
 
                if result:
                    pred = result["prediction"]
                    conf_pct = pred["confidence"] * 100
 
                    st.subheader("Prediction")
                    m1, m2 = st.columns(2)
                    m1.metric("Predicted class", pred["predicted_class"].upper())
                    m2.metric("Confidence", f"{conf_pct:.1f}%")
                    st.json(pred["all_probabilities"])
 
                    st.subheader("Retrieved Sources")
                    if len(result["retrieved_chunks"]) > 0:
                        st.dataframe(
                            result["retrieved_chunks"][
                                ["category", "source_file", "page_number", "distance"]
                            ],
                            use_container_width=True,
                        )
                    else:
                        st.info("No supporting literature retrieved for this case.")
 
                    st.subheader("Structured Report")
                    rd = result["report_data"]
                    st.markdown(f"**Clinical History**\n\n{rd['clinical_history']}")
                    st.markdown(f"**Technique**\n\n{rd['technique']}")
                    st.markdown(f"**Findings**\n\n{rd['findings']}")
                    st.markdown(f"**Impression**\n\n{rd['impression']}")
                    st.markdown(f"**Recommendations**\n\n{rd['recommendations']}")
 
                    if result["pdf_path"]:
                        st.subheader("PDF Report")
                        with open(result["pdf_path"], "rb") as f:
                            st.download_button(
                                "Download PDF Report",
                                f,
                                file_name=os.path.basename(result["pdf_path"]),
                                mime="application/pdf",
                                use_container_width=True,
                            )
        else:
            st.info("Upload an MRI image and click **Generate Report** to begin.")
 
 
# --------------------------------------------------------------------------
# Tab 2: Free-text follow-up Q&A
# --------------------------------------------------------------------------
 
with tab_ask:
    st.caption(
        "If you generated a report above, this question will be answered using that "
        "patient's prediction as the primary basis. Otherwise it's answered generally."
    )
 
    question = st.text_input(
        "Your question",
        placeholder="e.g. Why is the model confidence only this high, and is this probability spread typical?",
    )
    ask_clicked = st.button("Ask", type="primary")
 
    if ask_clicked:
        if not api_key:
            st.error("Please enter your OpenRouter API key in the sidebar first.")
        elif not question.strip():
            st.error("Please enter a question.")
        else:
            collection = _load_collection()
            prior_prediction = None
            last_result = st.session_state.get("last_result")
            if last_result:
                prior_prediction = last_result["prediction"]
                st.caption(
                    f"Using prediction from the last generated report: "
                    f"**{prior_prediction['predicted_class'].upper()}** "
                    f"({prior_prediction['confidence'] * 100:.1f}% confidence)"
                )
 
            with st.spinner("Retrieving literature and drafting an answer..."):
                try:
                    qa_result = ask_question(
                        collection=collection,
                        api_key=api_key,
                        question=question,
                        prediction=prior_prediction,
                        top_k=top_k,
                    )
                except Exception as e:
                    st.error(f"Question answering failed: {e}")
                    qa_result = None
 
            if qa_result:
                st.subheader("Retrieved Sources")
                if len(qa_result["retrieved_chunks"]) > 0:
                    st.dataframe(
                        qa_result["retrieved_chunks"][
                            ["category", "source_file", "page_number", "distance"]
                        ],
                        use_container_width=True,
                    )
                else:
                    st.info("No supporting literature retrieved for this question.")
 
                st.subheader("Answer")
                st.markdown(qa_result["answer"])
 
