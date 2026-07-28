"""
app.py - FastAPI Backend for Brain Tumor MRI Detection + MindScanDB.

Endpoints:
    GET  /                 → health check
    GET  /health           → detailed status (model loaded? DB reachable?)
    POST /predict          → upload MRI image → run AI → save diagnosis → return JSON

    GET  /patients         → list all patients
    GET  /patients/{id}    → one patient + their diagnosis history
    POST /patients         → add a new patient (the "Add Patient" button)

    GET  /doctors          → list all doctors
    POST /doctors          → add a new doctor

    GET  /diagnoses        → list all diagnoses (optionally filter by patient_id)
    GET  /dashboard        → summary stats for a dashboard view

Run with:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import pyodbc
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import model as brain_model
import utils
from config import settings
from database import check_connection, get_db, row_to_dict

# ─────────────────────────────────────────────────────────────────────────────
# Schema note (confirmed against the real MindScanDB via SSMS):
#
#   Doctor    (DoctorID, FirstName, LastName, Specialization, Phone, Email,
#              Role, SupervisorID)
#   Patient   (PatientID, DoctorID, FirstName, LastName, Age, Gender, Phone,
#              Email, CreatedAt)
#   Diagnosis (DiagnosisID, PatientID, Result, TumorType, ConfidenceScore,
#              MRIImagePath, Report, DiagnosisDate)
#
# Diagnosis has no DoctorID column of its own -- the doctor is reached via
# Patient.DoctorID. MRIImagePath is a Cloudinary URL; this backend does not
# upload to Cloudinary itself, so /predict accepts an optional `image_url`
# the frontend can pass in if it already uploaded the scan.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Startup / Shutdown: load model once
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Brain Tumor Detection API starting...")
    try:
        brain_model.load_model()
        print("[startup] Model ready! Accepting requests.")
    except FileNotFoundError as exc:
        print(f"[startup] WARNING: {exc}")
    yield
    print("[shutdown] Server stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Brain Tumor Detection API",
    description="Classifies brain MRI scans and stores results in MindScanDB.",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS — real origins only (never combine "*" with allow_credentials=True;
# browsers reject that combination anyway). Configure via ALLOWED_ORIGINS
# in your .env / environment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "message": "Brain Tumor Detection API is running!", "docs": "/docs"}


@app.get("/health")
async def health():
    model_ready = brain_model._model is not None
    db_ready = check_connection()
    overall = "ok" if (model_ready and db_ready) else "degraded"
    return {
        "status": overall,
        "model_ready": model_ready,
        "model_classes": brain_model.CLASS_NAMES if model_ready else None,
        "db_ready": db_ready,
        "message": (
            "Model and database ready."
            if overall == "ok"
            else "One or more dependencies not ready — check server logs."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Predict → saves a Diagnosis row tied to a patient (and optionally a doctor)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Brain MRI image (JPEG/PNG)"),
    patient_id: int = Form(..., description="Existing PatientID this scan belongs to"),
    image_url: Optional[str] = Form(
        None, description="Cloudinary URL, if the frontend already uploaded the scan"
    ),
    db: pyodbc.Cursor = Depends(get_db),
):
    if brain_model._model is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "model_not_ready", "message": "Model is not loaded. Check server logs."},
        )

    try:
        utils.validate_image_format(file.content_type, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": "invalid_format", "message": str(exc)})

    raw_bytes = await file.read()

    t_start = time.perf_counter()
    try:
        batch, orig_size = utils.preprocess_bytes(raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": "preprocessing_failed", "message": str(exc)})

    try:
        result = brain_model.predict(batch)
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": "prediction_failed", "message": str(exc)})

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

    # Confirm the patient exists before writing a Diagnosis row referencing it.
    db.execute("SELECT PatientID FROM Patient WHERE PatientID = ?", patient_id)
    if db.fetchone() is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "patient_not_found", "message": f"No patient with PatientID={patient_id}."},
        )

    # This 3-class model only predicts tumor subtypes, so every row is a
    # "Positive" result — mirrors the existing data in Diagnosis.
    report_text = (
        f"AI analysis suggests {result['display_name']} tumor with "
        f"{'high' if result['confidence_pct'] >= 90 else 'moderate'} confidence "
        f"({result['confidence_pct']}%)."
    )

    try:
        db.execute(
            """
            INSERT INTO Diagnosis
                (PatientID, Result, TumorType, ConfidenceScore, MRIImagePath, Report, DiagnosisDate)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            patient_id,
            "Positive",
            result["prediction"],
            result["confidence_pct"],
            image_url,
            report_text,
            datetime.utcnow(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": "db_write_failed", "message": str(exc)})

    return JSONResponse(content={
        **result,
        "patient_id": patient_id,
        "image_url": image_url,
        "report": report_text,
        "image_info": {
            "filename": file.filename or "unknown",
            "original_size": list(orig_size),
            "model_input_size": utils.IMG_SIZE,
        },
        "processing_time_ms": elapsed_ms,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Patients
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/patients")
def get_patients(db: pyodbc.Cursor = Depends(get_db)):
    db.execute(
        """
        SELECT PatientID, DoctorID, FirstName, LastName, Age, Gender, Phone, Email, CreatedAt
        FROM Patient
        ORDER BY CreatedAt DESC
        """
    )
    return [row_to_dict(db, row) for row in db.fetchall()]


@app.get("/patients/{patient_id}")
def get_patient(patient_id: int, db: pyodbc.Cursor = Depends(get_db)):
    db.execute(
        """
        SELECT PatientID, DoctorID, FirstName, LastName, Age, Gender, Phone, Email, CreatedAt
        FROM Patient WHERE PatientID = ?
        """,
        patient_id,
    )
    row = db.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Patient not found."})
    patient = row_to_dict(db, row)

    db.execute(
        """
        SELECT DiagnosisID, Result, TumorType, ConfidenceScore, MRIImagePath, Report, DiagnosisDate
        FROM Diagnosis WHERE PatientID = ? ORDER BY DiagnosisDate DESC
        """,
        patient_id,
    )
    patient["diagnoses"] = [row_to_dict(db, r) for r in db.fetchall()]
    return patient


@app.post("/patients", status_code=201)
def create_patient(
    first_name: str = Form(...),
    last_name: str = Form(...),
    age: Optional[int] = Form(None),
    gender: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    doctor_id: Optional[int] = Form(None),
    db: pyodbc.Cursor = Depends(get_db),
):
    db.execute(
        """
        INSERT INTO Patient (DoctorID, FirstName, LastName, Age, Gender, Phone, Email, CreatedAt)
        OUTPUT INSERTED.PatientID
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        doctor_id, first_name, last_name, age, gender, phone, email, datetime.utcnow(),
    )
    new_id = db.fetchone()[0]
    return {"patient_id": new_id, "first_name": first_name, "last_name": last_name, "message": "Patient created."}


# ─────────────────────────────────────────────────────────────────────────────
# Doctors
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/doctors")
def get_doctors(db: pyodbc.Cursor = Depends(get_db)):
    db.execute(
        """
        SELECT DoctorID, FirstName, LastName, Specialization, Phone, Email, Role, SupervisorID
        FROM Doctor ORDER BY LastName, FirstName
        """
    )
    return [row_to_dict(db, row) for row in db.fetchall()]


@app.post("/doctors", status_code=201)
def create_doctor(
    first_name: str = Form(...),
    last_name: str = Form(...),
    specialization: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    role: str = Form("Doctor"),
    supervisor_id: Optional[int] = Form(None),
    db: pyodbc.Cursor = Depends(get_db),
):
    db.execute(
        """
        INSERT INTO Doctor (FirstName, LastName, Specialization, Phone, Email, Role, SupervisorID)
        OUTPUT INSERTED.DoctorID
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        first_name, last_name, specialization, phone, email, role, supervisor_id,
    )
    new_id = db.fetchone()[0]
    return {"doctor_id": new_id, "first_name": first_name, "last_name": last_name, "message": "Doctor created."}


# ─────────────────────────────────────────────────────────────────────────────
# Diagnoses
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/diagnoses")
def get_diagnoses(patient_id: Optional[int] = None, db: pyodbc.Cursor = Depends(get_db)):
    if patient_id is not None:
        db.execute(
            """
            SELECT DiagnosisID, PatientID, Result, TumorType, ConfidenceScore, MRIImagePath, Report, DiagnosisDate
            FROM Diagnosis WHERE PatientID = ? ORDER BY DiagnosisDate DESC
            """,
            patient_id,
        )
    else:
        db.execute(
            """
            SELECT DiagnosisID, PatientID, Result, TumorType, ConfidenceScore, MRIImagePath, Report, DiagnosisDate
            FROM Diagnosis ORDER BY DiagnosisDate DESC
            """
        )
    return [row_to_dict(db, row) for row in db.fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/dashboard")
def get_dashboard(db: pyodbc.Cursor = Depends(get_db)):
    db.execute("SELECT COUNT(*) FROM Patient")
    total_patients = db.fetchone()[0]

    db.execute("SELECT COUNT(*) FROM Doctor")
    total_doctors = db.fetchone()[0]

    db.execute("SELECT COUNT(*) FROM Diagnosis")
    total_diagnoses = db.fetchone()[0]

    db.execute("SELECT TumorType, COUNT(*) AS cnt FROM Diagnosis GROUP BY TumorType")
    by_type = {row.TumorType: row.cnt for row in db.fetchall()}

    return {
        "total_patients": total_patients,
        "total_doctors": total_doctors,
        "total_diagnoses": total_diagnoses,
        "diagnoses_by_type": by_type,
    }
