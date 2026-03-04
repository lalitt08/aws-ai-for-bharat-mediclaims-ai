"""
Post-Submission Appeals Dashboard - Main Application
FastAPI server for the appeals management demo system.

This system reads denied claims produced by the Pre-Submission Agentic Pipeline
(claim_status.json + patients CSV) and presents them for appeal management.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import json
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import our API modules
try:
    from api.appeals_api import appeals_router
    from api.denial_analysis_api import denial_router
    from api.compliance_api import compliance_router
    from api.metrics_api import metrics_router

    # Import services
    from services.era_processor import ERAProcessor
    from services.denial_classifier import DenialClassifier
    from services.appeal_generator import AppealGenerator
    from services.compliance_checker import ComplianceChecker
except ImportError as e:
    print(f"Import error: {e}")
    print("Running in fallback mode with basic functionality")
    appeals_router = None
    denial_router = None
    compliance_router = None
    metrics_router = None

app = FastAPI(
    title="MediClaims AI - Post-Submission Appeals Dashboard",
    description="Intelligent appeals management and denial processing system",
    version="1.0.0"
)

# Allow the pre-submission dashboard to call our APIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Shared data paths (bridge between pre-submission and post-submission) ──
# The pre-submission pipeline writes claim_status.json and patients CSV here
PRE_SUBMISSION_DIR = os.path.dirname(BASE_DIR)  # parent = alpha project root
CLAIM_STATUS_FILE = os.path.join(PRE_SUBMISSION_DIR, "data", "claim_status.json")
PATIENTS_CSV_FILE = os.path.join(PRE_SUBMISSION_DIR, "data", "patients1.csv")


def _load_claim_statuses() -> Dict[str, Any]:
    """Load claim statuses written by the pre-submission pipeline."""
    try:
        if os.path.exists(CLAIM_STATUS_FILE):
            with open(CLAIM_STATUS_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: could not read claim_status.json: {e}")
    return {}


def _load_patients_csv() -> Dict[str, Dict[str, Any]]:
    """Load patient master data from the shared CSV. Returns dict keyed by patient_id."""
    patients: Dict[str, Dict[str, Any]] = {}
    try:
        if os.path.exists(PATIENTS_CSV_FILE):
            with open(PATIENTS_CSV_FILE, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pid = row.get("patient_id", "").strip()
                    if pid:
                        patients[pid] = row
    except Exception as e:
        print(f"Warning: could not read patients CSV: {e}")
    return patients


# ── Denial-code knowledge base (maps denial reasons to CO codes) ──
DENIAL_CODE_MAP = {
    "missing clinical documentation": {"code": "CO-16", "category": "documentation"},
    "incomplete provider credentials": {"code": "CO-16", "category": "documentation"},
    "diagnosis code mismatch": {"code": "CO-4", "category": "coding_error"},
    "modifier usage error": {"code": "CO-4", "category": "coding_error"},
    "prior authorization expired": {"code": "CO-197", "category": "prior_authorization"},
    "service level mismatch": {"code": "CO-197", "category": "prior_authorization"},
    "medical necessity not established": {"code": "CO-50", "category": "medical_necessity"},
    "timely filing limit exceeded": {"code": "CO-29", "category": "timely_filing"},
    "duplicate claim": {"code": "CO-18", "category": "duplicate_claim"},
}


def _map_denial_code(denial_reason: str) -> Dict[str, str]:
    """Map a free-text denial reason to a structured CO code."""
    reason_lower = (denial_reason or "").lower()
    for key, val in DENIAL_CODE_MAP.items():
        if key in reason_lower:
            return val
    return {"code": "CO-16", "category": "documentation"}

# Mount static files (create directory if it doesn't exist)
import shutil
static_dir = os.path.join(BASE_DIR, "frontend", "static")
frontend_dir = os.path.join(BASE_DIR, "frontend")

if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
    
# Copy frontend files to static directory if they don't exist
files_to_copy = [
    ("styles.css", "styles.css"),
    ("styles_new.css", "styles_new.css"),
    ("dashboard.js", "dashboard.js"),
    ("dashboard_simple.js", "dashboard_simple.js"),
    ("patients.js", "patients.js"),
    ("patient-details.js", "patient-details.js"),
    ("patient-details.css", "patient-details.css"),
    ("corrections.js", "corrections.js"),
    ("corrections.css", "corrections.css"),
]

for src_name, dst_name in files_to_copy:
    src_path = os.path.join(frontend_dir, src_name)
    dst_path = os.path.join(static_dir, dst_name)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setup templates
templates = Jinja2Templates(directory=frontend_dir)

# Include API routers (all now use real data from claim_status.json)
if appeals_router:
    app.include_router(appeals_router, prefix="/api/appeals", tags=["appeals"])
if denial_router:
    app.include_router(denial_router, prefix="/api/denials", tags=["denials"])
if compliance_router:
    app.include_router(compliance_router, prefix="/api/compliance", tags=["compliance"])
if metrics_router:
    app.include_router(metrics_router, prefix="/api/metrics", tags=["metrics"])

# Initialize services if available
try:
    era_processor = ERAProcessor()
    denial_classifier = DenialClassifier()
    appeal_generator = AppealGenerator()
    compliance_checker = ComplianceChecker()
except:
    era_processor = None
    denial_classifier = None
    appeal_generator = None
    compliance_checker = None

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main patients dashboard"""
    with open(os.path.join(frontend_dir, "index.html"), "r", encoding="utf-8") as file:
        return file.read()

@app.get("/patient-details/{patient_id}", response_class=HTMLResponse)
async def patient_details(patient_id: str):
    """Serve the patient details page"""
    with open(os.path.join(frontend_dir, "patient-details.html"), "r", encoding="utf-8") as file:
        return file.read()

@app.get("/corrections/{patient_id}", response_class=HTMLResponse)
async def corrections_page(patient_id: str):
    """Serve the corrections page"""
    try:
        with open(os.path.join(frontend_dir, "corrections.html"), "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "<html><body><h1>Corrections page coming soon...</h1><a href='javascript:history.back()'>Go Back</a></body></html>"

@app.get("/appeals-dashboard", response_class=HTMLResponse)
async def appeals_dashboard(request: Request):
    """Appeals management dashboard - redirect to main"""
    with open(os.path.join(frontend_dir, "index.html"), "r", encoding="utf-8") as file:
        return file.read()

@app.get("/era-processing", response_class=HTMLResponse)
async def era_processing_page(request: Request):
    """ERA Processing and Analysis page"""
    template_path = os.path.join(frontend_dir, "era-processing.html")
    if os.path.exists(template_path):
        return templates.TemplateResponse("era-processing.html", {"request": request})
    return HTMLResponse("<html><body><h1>ERA Processing</h1><p>Coming soon.</p><a href='/'>Back</a></body></html>")

@app.get("/appeal-detail/{appeal_id}", response_class=HTMLResponse)
async def appeal_detail_page(request: Request, appeal_id: str):
    """Individual appeal detail analysis page"""
    template_path = os.path.join(frontend_dir, "appeal-detail.html")
    if os.path.exists(template_path):
        return templates.TemplateResponse("appeal-detail.html", {"request": request, "appeal_id": appeal_id})
    return HTMLResponse(f"<html><body><h1>Appeal Detail: {appeal_id}</h1><p>Coming soon.</p><a href='/'>Back</a></body></html>")

@app.get("/pre-submission", response_class=HTMLResponse)
async def pre_submission_page(request: Request):
    """Pre-submission appeal analysis page"""
    template_path = os.path.join(frontend_dir, "pre-submission.html")
    if os.path.exists(template_path):
        return templates.TemplateResponse("pre-submission.html", {"request": request})
    return HTMLResponse("<html><body><h1>Pre-Submission Analysis</h1><p>Use the <a href='http://localhost:5000'>Pre-Submission Dashboard</a> to submit claims.</p></body></html>")

@app.get("/compliance", response_class=HTMLResponse)
async def compliance_page(request: Request):
    """Compliance and regulations dashboard"""
    template_path = os.path.join(frontend_dir, "compliance.html")
    if os.path.exists(template_path):
        return templates.TemplateResponse("compliance.html", {"request": request})
    return HTMLResponse("<html><body><h1>Compliance Dashboard</h1><p>Coming soon.</p><a href='/'>Back</a></body></html>")

@app.get("/metrics", response_class=HTMLResponse)
async def metrics_dashboard(request: Request):
    """Metrics and analytics dashboard"""
    template_path = os.path.join(frontend_dir, "metrics-dashboard.html")
    if os.path.exists(template_path):
        return templates.TemplateResponse("metrics-dashboard.html", {"request": request})
    return HTMLResponse("<html><body><h1>Metrics Dashboard</h1><p>Coming soon.</p><a href='/'>Back</a></body></html>")

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_dashboard(request: Request):
    """Executive analytics dashboard"""
    analytics_path = os.path.join(static_dir, "analytics-dashboard.html")
    if os.path.exists(analytics_path):
        return FileResponse(analytics_path)
    return HTMLResponse("<html><body><h1>Analytics Dashboard</h1><p>Coming soon.</p><a href='/'>Back</a></body></html>")

# ── Real data API endpoints (reads from pre-submission pipeline output) ──

@app.get("/api/denied-claims")
async def get_denied_claims():
    """Return denied claims produced by the pre-submission agentic pipeline.

    Merges claim_status.json (written by ClaimFlow) with the patients CSV
    so the post-submission dashboard shows real data, not mock data.
    """
    statuses = _load_claim_statuses()
    patients_csv = _load_patients_csv()

    denied_claims: List[Dict[str, Any]] = []
    for patient_id, entry in statuses.items():
        status = (entry.get("status") or "").lower()
        # Include anything that was denied / rejected / appealed / resubmitted
        if status not in ("approved", "learning_complete", "unknown", ""):
            csv_row = patients_csv.get(patient_id, {})
            submission = entry.get("submission_result") or {}
            denial_info = submission.get("denial_info") or {}
            denial_reason = denial_info.get("reason") or submission.get("message", "Claim denied")
            code_info = _map_denial_code(denial_reason)

            claim_amount = 0
            try:
                claim_amount = float(csv_row.get("claim_amount", 0) or entry.get("claim_amount", 0))
            except (ValueError, TypeError):
                pass

            success_rate = denial_info.get("success_rate", 0.75)
            try:
                success_pct = int(float(success_rate) * 100) if float(success_rate) <= 1 else int(float(success_rate))
            except (ValueError, TypeError):
                success_pct = 75

            denied_claims.append({
                "id": patient_id,
                "name": csv_row.get("name", f"Patient {patient_id}"),
                "age": int(csv_row.get("age", 0)) if csv_row.get("age") else 0,
                "claimId": entry.get("claim_id", ""),
                "amount": claim_amount,
                "payer": (csv_row.get("insurer") or "Unknown").lower().replace("bluecross", "bluecross").replace("blue cross", "bluecross"),
                "payerName": csv_row.get("insurer") or "Unknown",
                "priority": "high" if (entry.get("risk_score") or 0) > 0.6 else ("medium" if (entry.get("risk_score") or 0) > 0.3 else "low"),
                "denialReason": denial_reason,
                "denialCode": code_info["code"],
                "denialCategory": code_info["category"],
                "procedure": f"{csv_row.get('procedure_code', 'N/A')} - {csv_row.get('diagnosis_code', '')}",
                "serviceDate": csv_row.get("service_date", ""),
                "doctorName": csv_row.get("provider", ""),
                "successProbability": success_pct,
                "riskScore": entry.get("risk_score", 0),
                "issuesCount": entry.get("issues_count", 0),
                "status": entry.get("status", "denied"),
                "timestamp": entry.get("timestamp", ""),
                "medicalHistory": csv_row.get("medical_history", ""),
                "medications": csv_row.get("medications", ""),
                "allergies": csv_row.get("allergies", ""),
                "priorAuth": csv_row.get("prior_authorization", ""),
            })

    # Sort by risk score descending (highest risk first)
    denied_claims.sort(key=lambda x: x.get("riskScore", 0), reverse=True)
    return {"denied_claims": denied_claims, "total": len(denied_claims)}


@app.get("/api/denied-claims/{patient_id}")
async def get_denied_claim_detail(patient_id: str):
    """Return detailed info for a single denied claim."""
    statuses = _load_claim_statuses()
    patients_csv = _load_patients_csv()

    entry = statuses.get(patient_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Claim not found")

    csv_row = patients_csv.get(patient_id, {})
    submission = entry.get("submission_result") or {}
    denial_info = submission.get("denial_info") or {}
    denial_reason = denial_info.get("reason") or submission.get("message", "Claim denied")
    code_info = _map_denial_code(denial_reason)

    claim_amount = 0
    try:
        claim_amount = float(csv_row.get("claim_amount", 0) or 0)
    except (ValueError, TypeError):
        pass

    success_rate = denial_info.get("success_rate", 0.75)
    try:
        success_pct = int(float(success_rate) * 100) if float(success_rate) <= 1 else int(float(success_rate))
    except (ValueError, TypeError):
        success_pct = 75

    required_items = denial_info.get("required_items", [])
    details = denial_info.get("details", "")

    return {
        "id": patient_id,
        "name": csv_row.get("name", f"Patient {patient_id}"),
        "age": int(csv_row.get("age", 0)) if csv_row.get("age") else 0,
        "gender": csv_row.get("gender", ""),
        "dob": csv_row.get("dob", ""),
        "phone": csv_row.get("phone", ""),
        "email": csv_row.get("email", ""),
        "address": csv_row.get("address", ""),
        "claimId": entry.get("claim_id", ""),
        "amount": claim_amount,
        "payer": (csv_row.get("insurer") or "Unknown").lower(),
        "payerName": csv_row.get("insurer") or "Unknown",
        "priority": "high" if (entry.get("risk_score") or 0) > 0.6 else ("medium" if (entry.get("risk_score") or 0) > 0.3 else "low"),
        "denialReason": denial_reason,
        "denialCode": code_info["code"],
        "denialCategory": code_info["category"],
        "denialDetails": details,
        "requiredItems": required_items,
        "procedure": csv_row.get("procedure_code", "N/A"),
        "diagnosisCode": csv_row.get("diagnosis_code", ""),
        "serviceDate": csv_row.get("service_date", ""),
        "doctorName": csv_row.get("provider", ""),
        "successProbability": success_pct,
        "riskScore": entry.get("risk_score", 0),
        "issuesCount": entry.get("issues_count", 0),
        "status": entry.get("status", "denied"),
        "timestamp": entry.get("timestamp", ""),
        "medicalHistory": csv_row.get("medical_history", ""),
        "medications": csv_row.get("medications", ""),
        "allergies": csv_row.get("allergies", ""),
        "priorAuth": csv_row.get("prior_authorization", ""),
    }


# Fallback API endpoints if routers are not available
@app.get("/api/appeals/")
async def get_appeals():
    """Get all appeals — reads from the shared claim status data."""
    statuses = _load_claim_statuses()
    patients_csv = _load_patients_csv()

    appeals = []
    for patient_id, entry in statuses.items():
        csv_row = patients_csv.get(patient_id, {})
        submission = entry.get("submission_result") or {}
        denial_info = submission.get("denial_info") or {}
        denial_reason = denial_info.get("reason") or submission.get("message", "")
        claim_amount = 0
        try:
            claim_amount = float(csv_row.get("claim_amount", 0) or 0)
        except (ValueError, TypeError):
            pass

        appeals.append({
            "id": f"APP-{patient_id}",
            "claim_id": entry.get("claim_id", ""),
            "patient_name": csv_row.get("name", f"Patient {patient_id}"),
            "status": entry.get("status", "unknown"),
            "submission_date": entry.get("timestamp", ""),
            "denial_reason": denial_reason,
            "amount": claim_amount,
            "success_probability": int(float(denial_info.get("success_rate", 0.75)) * 100) if denial_info.get("success_rate") else 75,
        })

    return {"appeals": appeals}

@app.get("/api/metrics/summary")
async def get_metrics_summary():
    """Get metrics summary — computed from real claim status data."""
    statuses = _load_claim_statuses()
    total = len(statuses)
    approved = sum(1 for e in statuses.values() if (e.get("status") or "").lower() == "approved")
    denied = sum(1 for e in statuses.values() if (e.get("status") or "").lower() in ("denied", "rejected", "appeal_resubmitted", "appeal_generated"))
    resubmitted = sum(1 for e in statuses.values() if "resubmit" in (e.get("status") or "").lower())
    pending = total - approved - denied

    success_rate = round((approved / total * 100), 1) if total > 0 else 0

    total_amount = 0
    for e in statuses.values():
        sr = e.get("submission_result") or {}
        try:
            total_amount += float(sr.get("approved_amount", 0) or 0)
        except (ValueError, TypeError):
            pass

    return {
        "total_claims": total,
        "approved_claims": approved,
        "denied_claims": denied,
        "pending_claims": pending,
        "resubmitted_claims": resubmitted,
        "success_rate": success_rate,
        "total_recovered": round(total_amount, 2),
    }

@app.get("/api/denial-analysis/top-reasons")
async def get_top_denial_reasons():
    """Get top denial reasons — aggregated from real claim data."""
    statuses = _load_claim_statuses()
    reason_counts: Dict[str, int] = {}
    for entry in statuses.values():
        submission = entry.get("submission_result") or {}
        denial_info = submission.get("denial_info") or {}
        reason = denial_info.get("reason") or ""
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    total = sum(reason_counts.values()) or 1
    sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)

    return {
        "denial_reasons": [
            {"reason": r, "count": c, "percentage": round(c / total * 100, 1)}
            for r, c in sorted_reasons
        ]
    }

@app.get("/favicon.ico")
async def favicon():
    """Serve favicon - returns SVG icon as response"""
    from fastapi.responses import Response
    
    # SVG favicon content
    svg_content = '''<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>
        <circle cx='50' cy='50' r='45' fill='#2563eb'/>
        <path d='M30 35h40v8H30zM25 50h50v8H25zM35 65h30v8H35z' fill='white'/>
        <circle cx='50' cy='30' r='8' fill='#10b981'/>
    </svg>'''
    
    return Response(content=svg_content, media_type="image/svg+xml")


# ── Service-backed endpoints (use real services for AI-powered analysis) ──

@app.post("/api/generate-appeal/{patient_id}")
async def generate_appeal_for_patient(patient_id: str):
    """Generate an appeal using the AppealGenerator service with real patient data."""
    statuses = _load_claim_statuses()
    patients_csv = _load_patients_csv()

    entry = statuses.get(patient_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Patient not found")

    csv_row = patients_csv.get(patient_id, {})
    sub = entry.get("submission_result") or {}
    denial_info = sub.get("denial_info") or {}
    reason = denial_info.get("reason") or "Unknown"
    code_info = _map_denial_code(reason)

    denial_data = {
        "claim_id": entry.get("claim_id", ""),
        "patient_name": csv_row.get("name", f"Patient {patient_id}"),
        "denial_code": code_info["code"],
        "denial_reason": reason,
        "denied_amount": float(csv_row.get("claim_amount", 0) or 0),
        "payer": csv_row.get("insurer", "Unknown"),
        "service_date": csv_row.get("service_date", ""),
        "procedure_code": csv_row.get("procedure_code", ""),
        "diagnosis": csv_row.get("diagnosis_code", ""),
    }

    if appeal_generator:
        # Use the real DenialClassifier to classify first
        classification = None
        if denial_classifier:
            classification = denial_classifier.classify_denial(reason, code_info["code"])

        appeal = appeal_generator.generate_appeal(denial_data, classification)
        return appeal

    # Fallback
    return {
        "appeal_id": f"APP-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "claim_id": entry.get("claim_id", ""),
        "status": "draft",
        "message": "Appeal generated (basic mode - services not available)",
        "denial_reason": reason,
        "required_items": denial_info.get("required_items", []),
    }


@app.get("/api/classify-denial/{patient_id}")
async def classify_denial_for_patient(patient_id: str):
    """Classify a denial using the DenialClassifier service."""
    statuses = _load_claim_statuses()
    entry = statuses.get(patient_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Patient not found")

    sub = entry.get("submission_result") or {}
    denial_info = sub.get("denial_info") or {}
    reason = denial_info.get("reason") or "Unknown"
    code_info = _map_denial_code(reason)

    if denial_classifier:
        classification = denial_classifier.classify_denial(reason, code_info["code"])
        return classification

    return {
        "primary_classification": {"category": code_info["category"], "confidence": 0.8},
        "appeal_strategy": code_info["category"],
        "expected_success_rate": denial_info.get("success_rate", 0.5),
    }


@app.get("/api/compliance-check/{patient_id}")
async def compliance_check_for_patient(patient_id: str):
    """Run compliance check using the ComplianceChecker service."""
    statuses = _load_claim_statuses()
    patients_csv = _load_patients_csv()

    entry = statuses.get(patient_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Patient not found")

    csv_row = patients_csv.get(patient_id, {})
    sub = entry.get("submission_result") or {}

    claim_data = {
        "claim_id": entry.get("claim_id", ""),
        "patient_name": csv_row.get("name", ""),
        "service_date": csv_row.get("service_date", ""),
        "payer": (csv_row.get("insurer") or "").lower(),
        "claim_amount": float(csv_row.get("claim_amount", 0) or 0),
        "procedure_code": csv_row.get("procedure_code", ""),
        "prior_authorization": csv_row.get("prior_authorization", ""),
    }

    if compliance_checker:
        return compliance_checker.check_claim_compliance(claim_data)

    return {
        "claim_id": entry.get("claim_id", ""),
        "compliance_status": "warning",
        "compliance_score": 0.7,
        "message": "Basic compliance check (full service not available)",
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.post("/api/process-era/{patient_id}")
async def process_era_for_patient(patient_id: str, request: Request):
    """Process an uploaded ERA/835 file using AI agents.

    Accepts the raw ERA text, runs it through ERAProcessor + DenialClassifier,
    and returns structured analysis that the frontend renders.
    """
    body = await request.json()
    era_text = body.get("era_content", "")
    filename = body.get("filename", "uploaded.835")

    if not era_text.strip():
        raise HTTPException(status_code=400, detail="ERA content is empty")

    # Step 1: Process ERA via ERAProcessor
    if era_processor:
        era_result = era_processor.process_era_file(era_text, filename)
    else:
        # Lightweight fallback parser
        era_result = _fallback_parse_era(era_text, filename)

    # Step 2: Classify denials via DenialClassifier
    classified = []
    if denial_classifier and era_result.get("denials_extracted"):
        classified = denial_classifier.batch_classify_denials(era_result["denials_extracted"])
    elif era_result.get("denials_extracted"):
        classified = era_result["denials_extracted"]

    # Step 3: Generate appeal suggestions for top denials
    appeal_suggestions = []
    top_denials = (classified or era_result.get("denials_extracted", []))[:3]
    for denial in top_denials:
        classification = denial.get("classification")
        denial_data = {
            "claim_id": denial.get("claim_id", ""),
            "denial_code": denial.get("denial_code", ""),
            "denial_reason": denial.get("denial_reason", ""),
            "denied_amount": denial.get("denied_amount", 0),
            "payer": denial.get("payer", "Unknown"),
        }
        if appeal_generator:
            appeal = appeal_generator.generate_appeal(denial_data, classification)
            appeal_suggestions.append({
                "claim_id": denial_data["claim_id"],
                "denial_code": denial_data["denial_code"],
                "strategy": appeal.get("appeal_strategy", ""),
                "success_probability": appeal.get("success_probability", 0),
                "estimated_time": appeal.get("estimated_completion_time", ""),
            })

    # Step 4: Build statistics
    stats = {}
    if era_processor:
        stats = era_processor.get_era_statistics(era_result)

    return {
        "status": "processed",
        "filename": filename,
        "summary": era_result.get("summary", {}),
        "denials_count": len(era_result.get("denials_extracted", [])),
        "classified_denials": classified[:5],  # top 5
        "appeal_suggestions": appeal_suggestions,
        "statistics": stats,
        "llm_analysis": era_result.get("llm_analysis"),  # Azure OpenAI insights
        "processed_at": datetime.now().isoformat(),
    }


def _fallback_parse_era(era_text: str, filename: str) -> dict:
    """Basic ERA parser when full service is unavailable."""
    lines = [l.strip() for l in era_text.replace("~", "\n").split("\n") if l.strip()]
    denials = []
    claim_id = ""
    amount = 0.0
    for line in lines:
        segs = line.split("*")
        if segs[0] == "CLP" and len(segs) > 3:
            claim_id = segs[1]
            try:
                amount = float(segs[3])
            except (ValueError, IndexError):
                pass
        if segs[0] == "CAS" and len(segs) > 3:
            code = f"CO-{segs[2]}" if len(segs) > 2 else "CO-16"
            try:
                denied_amt = float(segs[3])
            except (ValueError, IndexError):
                denied_amt = amount
            denials.append({
                "claim_id": claim_id,
                "denial_code": code,
                "denial_reason": f"Denial code {code}",
                "denied_amount": denied_amt,
                "payer": "Unknown",
            })
    total = max(len(lines) // 5, len(denials) + 1)
    return {
        "file_id": f"ERA-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "filename": filename,
        "processed_at": datetime.now().isoformat(),
        "status": "completed",
        "summary": {
            "total_claims": total,
            "paid_claims": total - len(denials),
            "denied_claims": len(denials),
            "pending_claims": 0,
        },
        "denials_extracted": denials,
    }

if __name__ == "__main__":
    print("🏥 Starting MediClaims AI Post-Submission Appeals Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8003")
    print("📋 Appeals management: http://localhost:8003/appeals-dashboard")
    print("📈 Metrics: http://localhost:8003/metrics")
    print("🔧 API docs: http://localhost:8003/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
    )
