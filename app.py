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


# ==========================================================================
# Theme / CSS
# ==========================================================================
PRIMARY = "#6D5AE0"
PRIMARY_DARK = "#5A48C9"
PRIMARY_LIGHT = "#EDE9FE"
SIDEBAR_BG = "#100B23"
SIDEBAR_BG_2 = "#170F30"
INK = "#1E1B3A"
MUTED = "#6B7280"
BORDER = "#E7E7F0"
PAGE_BG = "#F4F5FA"
GREEN = "#16A34A"

NAV_ITEMS = [
    ("Dashboard", "🏠"),
    ("Generate Report", "📄"),
    ("Follow-up Q&A", "💬"),
    ("Literature Explorer", "📚"),
    ("How MindScan Works", "❓"),
]

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "last_result" not in st.session_state:
    st.session_state.last_result = None

active_index = 0
for _i, (_label, _icon) in enumerate(NAV_ITEMS):
    if _label == st.session_state.page or (
        st.session_state.page == "Dashboard" and _label in ("Generate Report", "Follow-up Q&A")
        and st.session_state.get("_nav_clicked") != _label
    ):
        pass
# Simple mapping: Dashboard / Generate Report / Follow-up Q&A all render the
# same combined dashboard view (as in the design). Only the last two nav
# items get their own placeholder pages.
DASHBOARD_ALIASES = {"Dashboard", "Generate Report", "Follow-up Q&A"}
try:
    active_index = [label for label, _ in NAV_ITEMS].index(st.session_state.page)
except ValueError:
    active_index = 0

st.markdown(
    f"""
    <style>
        html, body, [class*="css"] {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
        }}
        .stApp {{
            background: {PAGE_BG};
        }}
        #MainMenu, header[data-testid="stHeader"], footer {{ visibility: hidden; height: 0; }}
        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }}

        /* ---------------- Sidebar ---------------- */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {SIDEBAR_BG} 0%, {SIDEBAR_BG_2} 100%);
            min-width: 260px !important;
            width: 260px !important;
        }}
        section[data-testid="stSidebar"] > div {{
            padding-top: 1.2rem;
        }}
        section[data-testid="stSidebar"] * {{
            color: #E7E5F5;
        }}
        .ms-logo {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 0 1.1rem 1.2rem 1.1rem;
            font-size: 1.25rem;
            font-weight: 800;
            color: #FFFFFF;
        }}
        .ms-logo span.icon {{
            font-size: 1.4rem;
        }}
        section[data-testid="stSidebar"] div.stButton {{
            padding: 0 0.9rem;
            margin-bottom: 4px;
        }}
        section[data-testid="stSidebar"] div.stButton > button {{
            width: 100%;
            text-align: left;
            background: transparent;
            border: none;
            border-radius: 10px;
            padding: 0.55rem 0.8rem;
            font-size: 0.92rem;
            font-weight: 500;
            color: #C7C4E0 !important;
            transition: background 0.15s ease;
        }}
        section[data-testid="stSidebar"] div.stButton > button:hover {{
            background: rgba(255,255,255,0.06);
            color: #FFFFFF !important;
        }}
        section[data-testid="stSidebar"] div.stButton:nth-of-type({active_index + 1}) > button {{
            background: {PRIMARY} !important;
            color: #FFFFFF !important;
            font-weight: 600;
            box-shadow: 0 4px 10px rgba(109, 90, 224, 0.45);
        }}
        .ms-sidebar-heading {{
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 0.7rem;
            color: #8A87AA;
            font-weight: 700;
            padding: 1.1rem 1.1rem 0.4rem 1.1rem;
        }}
        .ms-sidebar-note {{
            margin: 0.6rem 1.1rem 0 1.1rem;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 0.85rem;
            font-size: 0.82rem;
            line-height: 1.4;
            color: #C7C4E0;
        }}
        .ms-research-mode {{
            margin: 1.4rem 1.1rem 0.4rem 1.1rem;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 0.75rem 0.85rem;
            font-size: 0.82rem;
            display: flex;
            gap: 8px;
            align-items: flex-start;
        }}
        .ms-research-mode .title {{ font-weight: 700; color: #FFFFFF; display:block; }}
        .ms-research-mode .sub {{ color: #9B98BC; font-size: 0.76rem; }}
        section[data-testid="stSidebar"] .stSlider label {{
            color: #E7E5F5 !important;
            font-size: 0.85rem;
            font-weight: 600;
        }}

        /* ---------------- Header ---------------- */
        .ms-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.4rem;
        }}
        .ms-title-row {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .ms-title-row .brain {{ font-size: 2.1rem; }}
        .ms-title-row h1 {{
            font-size: 1.9rem;
            font-weight: 800;
            color: {INK};
            margin: 0;
            line-height: 1.1;
        }}
        .ms-title-row .sub {{
            color: {MUTED};
            font-size: 0.92rem;
            margin-top: 2px;
        }}
        .ms-status-card {{
            background: #FFFFFF;
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 0.6rem 1rem;
            box-shadow: 0 2px 6px rgba(20,10,60,0.05);
            font-size: 0.82rem;
            min-width: 210px;
        }}
        .ms-status-card .row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }}
        .ms-status-card .row:last-child {{ margin-bottom: 0; }}
        .ms-status-card .label {{ color: {MUTED}; }}
        .dot {{
            display: inline-block;
            width: 8px; height: 8px;
            border-radius: 50%;
            background: {GREEN};
            margin-right: 5px;
            box-shadow: 0 0 0 3px rgba(22,163,74,0.15);
        }}

        /* ---------------- Cards ---------------- */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: #FFFFFF;
            border: 1px solid {BORDER} !important;
            border-radius: 14px !important;
            box-shadow: 0 2px 10px rgba(20,10,60,0.05);
            padding: 0.3rem 0.2rem;
        }}
        .ms-card-title {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 700;
            color: {INK};
            font-size: 1.02rem;
            margin-bottom: 0.9rem;
        }}
        .ms-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 22px; height: 22px;
            border-radius: 50%;
            background: {PRIMARY};
            color: #fff;
            font-size: 0.75rem;
            font-weight: 700;
            flex: none;
        }}

        /* ---------------- Uploader ---------------- */
        [data-testid="stFileUploaderDropzone"] {{
            background: #FAFAFE;
            border: 1.5px dashed #C9C6E8 !important;
            border-radius: 12px;
        }}
        [data-testid="stFileUploaderDropzoneInstructions"] svg {{ color: {PRIMARY}; }}

        /* ---------------- Text inputs ---------------- */
        .stTextInput input, .stTextArea textarea, .stNumberInput input {{
            border-radius: 10px !important;
            border: 1px solid {BORDER} !important;
        }}
        label, .stTextInput label, .stTextArea label {{
            font-weight: 600 !important;
            color: {INK} !important;
            font-size: 0.85rem !important;
        }}

        /* ---------------- Buttons ---------------- */
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
            border: none;
            border-radius: 10px;
            font-weight: 700;
            padding: 0.6rem 1rem;
            box-shadow: 0 4px 10px rgba(109,90,224,0.35);
        }}
        .stButton > button[kind="secondary"] {{
            border-radius: 10px;
            border: 1px solid {PRIMARY};
            color: {PRIMARY};
            font-weight: 600;
        }}

        /* ---------------- Prediction block ---------------- */
        .ms-pred-label {{
            font-size: 0.78rem;
            font-weight: 700;
            color: {MUTED};
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 2px;
        }}
        .ms-pred-value {{
            font-size: 1.5rem;
            font-weight: 800;
            color: {PRIMARY};
            margin-bottom: 2px;
        }}
        .ms-pred-sub {{
            color: {MUTED};
            font-size: 0.85rem;
            margin-bottom: 1rem;
        }}
        .ms-conf-value {{
            font-size: 1.6rem;
            font-weight: 800;
            color: {GREEN};
            margin: 2px 0 6px 0;
        }}
        .ms-progress-track {{
            width: 100%;
            height: 9px;
            background: {PRIMARY_LIGHT};
            border-radius: 6px;
            overflow: hidden;
            margin-bottom: 4px;
        }}
        .ms-progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
            border-radius: 6px;
        }}
        .ms-progress-scale {{
            display: flex;
            justify-content: space-between;
            font-size: 0.72rem;
            color: {MUTED};
            margin-bottom: 1rem;
        }}
        .ms-finding {{
            display: flex;
            align-items: flex-start;
            gap: 8px;
            font-size: 0.88rem;
            color: {INK};
            margin-bottom: 6px;
        }}
        .ms-finding .check {{
            color: {GREEN};
            font-weight: 800;
        }}

        /* ---------------- Tabs (report sections) ---------------- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 1.4rem;
            border-bottom: 1px solid {BORDER};
        }}
        .stTabs [data-baseweb="tab"] {{
            padding: 0 0 0.6rem 0;
            font-weight: 600;
            color: {MUTED};
        }}
        .stTabs [aria-selected="true"] {{
            color: {PRIMARY} !important;
            border-bottom-color: {PRIMARY} !important;
        }}

        .ms-section-heading {{
            color: {PRIMARY};
            font-weight: 700;
            font-size: 0.92rem;
            margin-bottom: 4px;
        }}
        .ms-body-text {{
            color: {INK};
            font-size: 0.92rem;
            line-height: 1.55;
        }}
        .ms-reco-panel {{
            background: {PRIMARY_LIGHT};
            border-radius: 12px;
            padding: 1rem 1.1rem;
        }}
        .ms-reco-panel ul {{ margin: 6px 0 0 1.1rem; padding: 0; }}
        .ms-reco-panel li {{ margin-bottom: 4px; font-size: 0.88rem; color: {INK}; }}

        .ms-example-chip {{
            display: inline-block;
            background: {PRIMARY_LIGHT};
            color: {PRIMARY_DARK};
            border-radius: 999px;
            padding: 5px 12px;
            font-size: 0.78rem;
            font-weight: 600;
            margin: 0 6px 6px 0;
        }}

        .ms-footer {{
            text-align: center;
            color: {MUTED};
            font-size: 0.8rem;
            margin-top: 1.6rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# Sidebar
# ==========================================================================
with st.sidebar:
    st.markdown(
        '<div class="ms-logo"><span class="icon">🧠</span> MindScan</div>',
        unsafe_allow_html=True,
    )

    for label, icon in NAV_ITEMS:
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label
            st.rerun()

    st.markdown('<div class="ms-sidebar-heading">App Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Number of literature sources", 1, 10, 5, label_visibility="visible")

    st.markdown(
        '<div class="ms-sidebar-note">Research/educational tool — not a medical diagnosis. '
        "All findings require clinician confirmation.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="ms-research-mode">🛡️<div><span class="title">Research Mode</span>'
        '<span class="sub">Not for clinical decision-making</span></div></div>',
        unsafe_allow_html=True,
    )


# ==========================================================================
# Header
# ==========================================================================
api_key = st.secrets.get("OPENROUTER_API_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")

h_left, h_right = st.columns([3, 1])
with h_left:
    st.markdown(
        """
        <div class="ms-title-row">
            <span class="brain">🧠</span>
            <div>
                <h1>MindScan</h1>
                <div class="sub">AI-Powered Brain Tumor Classification + Clinical Report Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with h_right:
    st.markdown(
        f"""
        <div class="ms-status-card">
            <div class="row"><span class="label">System Status</span>
                <span><span class="dot"></span>Online</span></div>
            <div class="row"><span class="label">Model: AdaptiveScan v1.0</span>
                <span class="dot"></span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not api_key:
    st.warning("Enter your OpenRouter API key below to generate reports or ask questions.")
    api_key = st.text_input(
        "OpenRouter API key",
        type="password",
        help="Get a free key at https://openrouter.ai/keys. Also enable "
        "'Free endpoints that may train on request data' at "
        "https://openrouter.ai/settings/privacy to use :free models.",
    )


# ==========================================================================
# Cached resources
# ==========================================================================
@st.cache_resource(show_spinner="Loading knowledge base...")
def _load_collection():
    return get_collection()


@st.cache_resource(show_spinner="Loading tumor-classification models...")
def _warm_up_model():
    from model_pipeline import load_pipeline
    return load_pipeline()


# ==========================================================================
# Placeholder pages for nav items that don't have a dedicated view
# ==========================================================================
if st.session_state.page == "Literature Explorer":
    with st.container(border=True):
        st.markdown(
            '<div class="ms-card-title"><span class="ms-badge">i</span> Literature Explorer</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="ms-body-text">Browse the literature knowledge base directly here. '
            "This view is a placeholder — wire it up to <code>vector_store.get_collection()</code> "
            "to list and search indexed sources.</div>",
            unsafe_allow_html=True,
        )
    st.stop()

if st.session_state.page == "How MindScan Works":
    with st.container(border=True):
        st.markdown(
            '<div class="ms-card-title"><span class="ms-badge">?</span> How MindScan Works</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="ms-body-text">MindScan classifies an uploaded MRI with a trained '
            "model, retrieves supporting literature from a vector knowledge base, and drafts "
            "a structured clinical report grounded in that literature. All outputs are for "
            "research/education only and require clinician confirmation.</div>",
            unsafe_allow_html=True,
        )
    st.stop()


# ==========================================================================
# Dashboard (Generate Report + Follow-up Q&A combined)
# ==========================================================================
col_left, col_right = st.columns([1, 1.6], gap="medium")

# ---------------- Left column: upload + patient context ----------------
with col_left:
    with st.container(border=True):
        st.markdown(
            '<div class="ms-card-title"><span class="ms-badge">1</span> Upload MRI Image</div>',
            unsafe_allow_html=True,
        )
        uploaded_image = st.file_uploader(
            "Drag & drop MRI image here, or click to browse. Supports JPG, PNG (max 200MB).",
            type=["jpg", "jpeg", "png"],
            key="report_image",
            label_visibility="collapsed",
        )

    with st.container(border=True):
        st.markdown(
            '<div class="ms-card-title"><span class="ms-badge">2</span> Patient Context (Optional)</div>',
            unsafe_allow_html=True,
        )
        patient_age = st.text_input("Patient Age", placeholder="e.g., 45", key="patient_age")
        clinical_notes = st.text_area(
            "Clinical Notes",
            placeholder="e.g., headache and visual disturbance for 3 weeks.",
            key="clinical_notes",
            height=110,
        )
        generate_clicked = st.button(
            "📄  Generate Clinical Report", type="primary", use_container_width=True
        )

patient_context_parts = []
if patient_age.strip():
    patient_context_parts.append(f"Patient age: {patient_age.strip()}.")
if clinical_notes.strip():
    patient_context_parts.append(clinical_notes.strip())
patient_context = " ".join(patient_context_parts)

if generate_clicked:
    if not api_key:
        st.error("Please enter your OpenRouter API key above first.")
    elif uploaded_image is None:
        st.error("Please upload an MRI image.")
    else:
        suffix = os.path.splitext(uploaded_image.name)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_image.getbuffer())
            tmp_path = tmp.name

        collection = _load_collection()

        with st.spinner("Classifying image, retrieving literature, and drafting report..."):
            try:
                result = generate_medical_report(
                    collection=collection,
                    api_key=api_key,
                    image_path=tmp_path,
                    patient_context=patient_context,
                    top_k=top_k,
                )
                result["_uploaded_image"] = uploaded_image
                st.session_state["last_result"] = result
            except Exception as e:
                st.error(f"Report generation failed: {e}")

last_result = st.session_state.get("last_result")

# ---------------- Right column: AI analysis results ----------------
with col_right:
    with st.container(border=True):
        st.markdown(
            '<div class="ms-card-title"><span class="ms-badge">3</span> AI Analysis Results</div>',
            unsafe_allow_html=True,
        )

        if not last_result:
            st.markdown(
                '<div class="ms-body-text" style="color:#6B7280;">Upload an MRI image and click '
                "<b>Generate Clinical Report</b> to see predictions here.</div>",
                unsafe_allow_html=True,
            )
        else:
            pred = last_result["prediction"]
            conf_pct = pred["confidence"] * 100
            rd = last_result["report_data"]

            res_left, res_right = st.columns([1, 1], gap="medium")

            with res_left:
                st.markdown('<div class="ms-pred-label">Predicted Diagnosis</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="ms-pred-value">{pred["predicted_class"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="ms-pred-sub">Model confidence score below</div>', unsafe_allow_html=True)

                st.markdown('<div class="ms-pred-label">Confidence Score</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ms-conf-value">{conf_pct:.0f}%</div>', unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="ms-progress-track">
                        <div class="ms-progress-fill" style="width:{min(conf_pct,100):.0f}%;"></div>
                    </div>
                    <div class="ms-progress-scale"><span>0%</span><span>50%</span><span>100%</span></div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="ms-pred-label">Key Findings</div>', unsafe_allow_html=True)
                findings_text = rd.get("findings", "")
                findings_items = [
                    s.strip() for s in findings_text.replace("\n", ". ").split(".") if s.strip()
                ][:4]
                if findings_items:
                    for item in findings_items:
                        st.markdown(
                            f'<div class="ms-finding"><span class="check">✓</span><span>{item}.</span></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        '<div class="ms-body-text" style="color:#6B7280;">No findings text returned.</div>',
                        unsafe_allow_html=True,
                    )

            with res_right:
                heatmap_path = last_result.get("heatmap_path") or last_result.get("heatmap")
                if heatmap_path:
                    st.markdown("**Tumor Localization (Heatmap)**")
                    st.image(heatmap_path, use_container_width=True)
                else:
                    st.markdown("**MRI Image**")
                    st.image(last_result["_uploaded_image"], use_container_width=True)
                    st.caption(
                        "No Grad-CAM heatmap was returned by the pipeline. Return a "
                        "`heatmap_path` from `generate_medical_report()` to show a localization "
                        "overlay here."
                    )

                with st.expander("Class probabilities"):
                    st.json(pred["all_probabilities"])

# ---------------- Clinical report draft ----------------
if last_result:
    with st.container(border=True):
        title_col, dl_col = st.columns([4, 1])
        with title_col:
            st.markdown(
                '<div class="ms-card-title"><span class="ms-badge">4</span> Clinical Report (Draft)</div>',
                unsafe_allow_html=True,
            )
        with dl_col:
            if last_result.get("pdf_path"):
                with open(last_result["pdf_path"], "rb") as f:
                    st.download_button(
                        "⬇ Download Report",
                        f,
                        file_name=os.path.basename(last_result["pdf_path"]),
                        mime="application/pdf",
                        use_container_width=True,
                    )

        rd = last_result["report_data"]
        tabs = st.tabs(["Summary", "Findings", "Impression", "Recommendations", "References"])

        with tabs[0]:
            st.markdown('<div class="ms-section-heading">Clinical Summary</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ms-body-text">{rd.get("clinical_history", "")}</div>', unsafe_allow_html=True)

        with tabs[1]:
            st.markdown('<div class="ms-section-heading">Findings</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ms-body-text">{rd.get("findings", "")}</div>', unsafe_allow_html=True)

        with tabs[2]:
            st.markdown('<div class="ms-section-heading">Impression</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ms-body-text">{rd.get("impression", "")}</div>', unsafe_allow_html=True)

        with tabs[3]:
            reco_items = [s.strip() for s in rd.get("recommendations", "").split("\n") if s.strip()]
            st.markdown('<div class="ms-reco-panel">', unsafe_allow_html=True)
            st.markdown('<div class="ms-section-heading">Recommendations</div>', unsafe_allow_html=True)
            if reco_items:
                st.markdown(
                    "<ul>" + "".join(f"<li>{item}</li>" for item in reco_items) + "</ul>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f'<div class="ms-body-text">{rd.get("recommendations", "")}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with tabs[4]:
            st.markdown('<div class="ms-section-heading">References</div>', unsafe_allow_html=True)
            chunks = last_result.get("retrieved_chunks")
            if chunks is not None and len(chunks) > 0:
                st.dataframe(
                    chunks[["category", "source_file", "page_number", "distance"]],
                    use_container_width=True,
                )
            else:
                st.markdown(
                    '<div class="ms-body-text" style="color:#6B7280;">No supporting literature '
                    "retrieved for this case.</div>",
                    unsafe_allow_html=True,
                )

st.write("")

# ---------------- Ask a follow-up question ----------------
with st.container(border=True):
    st.markdown(
        '<div class="ms-card-title"><span class="ms-badge">5</span> Ask a Follow-up Question</div>',
        unsafe_allow_html=True,
    )
    if last_result:
        pp = last_result["prediction"]
        st.caption(
            f"Using prediction from the last generated report: "
            f"**{pp['predicted_class'].upper()}** ({pp['confidence'] * 100:.1f}% confidence)"
        )
    else:
        st.caption("Generate a report above, or ask a general question below.")

    q_col, btn_col, ex_col = st.columns([2.2, 0.5, 1.3])
    with q_col:
        question = st.text_input(
            "Your question",
            placeholder="Type your question here...",
            key="follow_up_question",
            label_visibility="collapsed",
        )
    with btn_col:
        ask_clicked = st.button("📨  Ask", type="primary", use_container_width=True)
    with ex_col:
        st.markdown(
            '<div style="font-size:0.78rem; font-weight:700; color:#6B7280; margin-bottom:4px;">Example Questions</div>'
            '<span class="ms-example-chip">What is the prognosis?</span>'
            '<span class="ms-example-chip">What are treatment options?</span>'
            '<span class="ms-example-chip">How does this compare to literature?</span>',
            unsafe_allow_html=True,
        )

    if ask_clicked:
        if not api_key:
            st.error("Please enter your OpenRouter API key above first.")
        elif not question.strip():
            st.error("Please enter a question.")
        else:
            collection = _load_collection()
            prior_prediction = last_result["prediction"] if last_result else None

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
                st.markdown('<div class="ms-section-heading">Answer</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ms-body-text">{qa_result["answer"]}</div>', unsafe_allow_html=True)

                with st.expander("Retrieved sources"):
                    chunks = qa_result.get("retrieved_chunks")
                    if chunks is not None and len(chunks) > 0:
                        st.dataframe(
                            chunks[["category", "source_file", "page_number", "distance"]],
                            use_container_width=True,
                        )
                    else:
                        st.info("No supporting literature retrieved for this question.")

st.markdown(
    '<div class="ms-footer">🔒 MindScan is a research/educational tool. All outputs must be '
    "reviewed and confirmed by a qualified healthcare professional.</div>",
    unsafe_allow_html=True,
)
 
