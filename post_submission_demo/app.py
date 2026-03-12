"""
Post-Submission Appeals Dashboard - Main Application
FastAPI server for the appeals management demo system.

This system reads denied claims produced by the Pre-Submission Agentic Pipeline
(claim_status.json + patients CSV) and presents them for appeal management.
Integrates with AWS Bedrock (Nova Micro) for AI-powered ERA analysis and appeal generation.
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

# ── AWS Bedrock integration ──────────────────────────────────────────────────
try:
    import boto3
    _bedrock = boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    BEDROCK_MODEL = "us.amazon.nova-micro-v1:0"
    BEDROCK_AVAILABLE = True
except Exception:
    _bedrock = None
    BEDROCK_AVAILABLE = False

def _bedrock_invoke(prompt: str, max_tokens: int = 800) -> str:
    """Call Bedrock Nova Micro and return the text response."""
    if not BEDROCK_AVAILABLE or not _bedrock:
        return ""
    try:
        body = json.dumps({
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxNewTokens": max_tokens, "temperature": 0.3},
        })
        resp = _bedrock.invoke_model(modelId=BEDROCK_MODEL, body=body)
        result = json.loads(resp["body"].read())
        return result["output"]["message"]["content"][0]["text"].strip()
    except Exception as e:
        print(f"Bedrock error: {e}")
        return ""

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
    version="1.0.0",
    root_path=""
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
    
# Always copy frontend files to static — ensures latest source is served
files_to_copy = [
    "styles.css", "styles_new.css",
    "dashboard.js", "dashboard_simple.js",
    "patients.js", "patient-details.js", "patient-details.css",
    "corrections.js", "corrections.css",
    "era.js", "era.css",
]

for fname in files_to_copy:
    src_path = os.path.join(frontend_dir, fname)
    if os.path.exists(src_path):
        shutil.copy2(src_path, os.path.join(static_dir, fname))

app.mount("/static", StaticFiles(directory=static_dir), name="static")
# Also serve static under /appeals/static for ALB path-based routing
app.mount("/appeals/static", StaticFiles(directory=static_dir), name="appeals_static")

# ── ALB path-prefix rewrite middleware ────────────────────────────────────────
# The ALB forwards /appeals/api/... as-is (no path stripping).
# This middleware rewrites /appeals/api/* → /api/* and
# /appeals/patient-details/* → /patient-details/* etc. so all existing
# routes work without duplication.
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

class StripAppealsPrefix(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        path = request.scope["path"]
        # Rewrite /appeals/api/* → /api/*
        # Rewrite /appeals/patient-details/* → /patient-details/*
        # Rewrite /appeals/corrections/* → /corrections/*
        # Rewrite /appeals/claim-journey/* → /claim-journey/*
        # Leave /appeals/ and /appeals alone (handled by explicit routes)
        prefixes = ["/appeals/api/", "/appeals/patient-details/",
                    "/appeals/corrections/", "/appeals/claim-journey/"]
        for prefix in prefixes:
            if path.startswith(prefix):
                new_path = path[len("/appeals"):]
                request.scope["path"] = new_path
                break
        return await call_next(request)

app.add_middleware(StripAppealsPrefix)

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

# ALB path-based routing: /appeals/ prefix aliases
@app.get("/appeals/", response_class=HTMLResponse)
@app.get("/appeals", response_class=HTMLResponse)
async def dashboard_appeals(request: Request):
    """Main patients dashboard (ALB /appeals/ prefix)"""
    with open(os.path.join(frontend_dir, "index.html"), "r", encoding="utf-8") as file:
        return file.read()

@app.get("/patient-details/{patient_id}", response_class=HTMLResponse)
async def patient_details(patient_id: str):
    """Serve the patient details page"""
    with open(os.path.join(frontend_dir, "patient-details.html"), "r", encoding="utf-8") as file:
        return file.read()

@app.get("/appeals/patient-details/{patient_id}", response_class=HTMLResponse)
async def patient_details_appeals(patient_id: str):
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

@app.get("/appeals/corrections/{patient_id}", response_class=HTMLResponse)
async def corrections_page_appeals(patient_id: str):
    """Serve the corrections page (ALB /appeals/ prefix)"""
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
    return HTMLResponse("<html><body><h1>Pre-Submission Analysis</h1><p>Use the <a href='/'>Pre-Submission Dashboard</a> to submit claims.</p></body></html>")

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

# Mock denied claims data - always available regardless of pre-submission state
MOCK_DENIED_CLAIMS = [
    {"id": "PAT002", "name": "Sarah Johnson", "age": 34, "claimId": "CLM-20260301-PAT002", "amount": 2450.00, "payer": "aetna", "payerName": "Aetna", "priority": "high", "denialReason": "Prior authorization required for procedure", "denialCode": "CO-197", "denialCategory": "prior_authorization", "procedure": "99215 - E11.9", "serviceDate": "2026-02-15", "doctorName": "Dr. Smith", "successProbability": 78, "riskScore": 0.72, "issuesCount": 2, "status": "denied"},
    {"id": "PAT004", "name": "Emma Wilson", "age": 45, "claimId": "CLM-20260228-PAT004", "amount": 3200.00, "payer": "bluecross", "payerName": "BlueCross", "priority": "urgent", "denialReason": "Medical necessity not established", "denialCode": "CO-50", "denialCategory": "medical_necessity", "procedure": "99214 - J45.9", "serviceDate": "2026-02-20", "doctorName": "Dr. Williams", "successProbability": 65, "riskScore": 0.85, "issuesCount": 3, "status": "denied"},
    {"id": "PAT007", "name": "Michael Chen", "age": 52, "claimId": "CLM-20260225-PAT007", "amount": 1875.50, "payer": "united", "payerName": "United Healthcare", "priority": "high", "denialReason": "Insufficient documentation provided", "denialCode": "CO-16", "denialCategory": "documentation", "procedure": "99213 - I10", "serviceDate": "2026-02-18", "doctorName": "Dr. Garcia", "successProbability": 82, "riskScore": 0.68, "issuesCount": 1, "status": "denied"},
    {"id": "PAT011", "name": "Kevin Anderson", "age": 38, "claimId": "CLM-20260222-PAT011", "amount": 4100.00, "payer": "cigna", "payerName": "Cigna", "priority": "medium", "denialReason": "Procedure code not covered under policy", "denialCode": "CO-96", "denialCategory": "policy_exclusion", "procedure": "99215 - M54.5", "serviceDate": "2026-02-12", "doctorName": "Dr. Martinez", "successProbability": 55, "riskScore": 0.45, "issuesCount": 2, "status": "denied"},
    {"id": "PAT015", "name": "Andrew Harris", "age": 61, "claimId": "CLM-20260220-PAT015", "amount": 5250.00, "payer": "aetna", "payerName": "Aetna", "priority": "urgent", "denialReason": "Duplicate claim submission detected", "denialCode": "CO-18", "denialCategory": "duplicate_claim", "procedure": "99214 - K21.0", "serviceDate": "2026-02-08", "doctorName": "Dr. Thompson", "successProbability": 88, "riskScore": 0.92, "issuesCount": 1, "status": "denied"},
    {"id": "PAT017", "name": "Matthew Lewis", "age": 29, "claimId": "CLM-20260218-PAT017", "amount": 1650.00, "payer": "bluecross", "payerName": "BlueCross", "priority": "medium", "denialReason": "Timely filing limit exceeded", "denialCode": "CO-29", "denialCategory": "timely_filing", "procedure": "99212 - R10.9", "serviceDate": "2026-01-25", "doctorName": "Dr. Robinson", "successProbability": 42, "riskScore": 0.38, "issuesCount": 1, "status": "denied"},
    {"id": "PAT020", "name": "Jennifer Clark", "age": 43, "claimId": "CLM-20260215-PAT020", "amount": 2890.00, "payer": "united", "payerName": "United Healthcare", "priority": "high", "denialReason": "Invalid diagnosis code combination", "denialCode": "CO-4", "denialCategory": "coding_error", "procedure": "99214 - G43.909", "serviceDate": "2026-02-05", "doctorName": "Dr. Lee", "successProbability": 71, "riskScore": 0.65, "issuesCount": 2, "status": "denied"},
    {"id": "PAT024", "name": "David Martinez", "age": 55, "claimId": "CLM-20260212-PAT024", "amount": 3750.00, "payer": "cigna", "payerName": "Cigna", "priority": "high", "denialReason": "Service not covered for patient age group", "denialCode": "CO-167", "denialCategory": "policy_exclusion", "procedure": "99215 - N18.3", "serviceDate": "2026-02-01", "doctorName": "Dr. Brown", "successProbability": 60, "riskScore": 0.58, "issuesCount": 2, "status": "denied"},
    {"id": "PAT028", "name": "Lisa Rodriguez", "age": 36, "claimId": "CLM-20260210-PAT028", "amount": 2100.00, "payer": "aetna", "payerName": "Aetna", "priority": "medium", "denialReason": "Missing modifier for procedure code", "denialCode": "CO-4", "denialCategory": "coding_error", "procedure": "99213 - F32.9", "serviceDate": "2026-01-28", "doctorName": "Dr. Taylor", "successProbability": 85, "riskScore": 0.42, "issuesCount": 1, "status": "denied"},
]

@app.get("/api/denied-claims")
@app.get("/appeals/api/denied-claims")
async def get_denied_claims():
    """Return denied claims - uses mock data for consistent demo experience."""
    # Return mock denied claims for demo - independent of pre-submission pipeline
    denied_claims = MOCK_DENIED_CLAIMS.copy()
    # Sort by risk score descending (highest risk first)
    denied_claims.sort(key=lambda x: x.get("riskScore", 0), reverse=True)
    return {"denied_claims": denied_claims, "total": len(denied_claims)}


@app.get("/api/denied-claims/{patient_id}")
@app.get("/appeals/api/denied-claims/{patient_id}")
async def get_denied_claim_detail(patient_id: str):
    """Return detailed info for a single denied claim - uses mock data."""
    # Find in mock data first
    for claim in MOCK_DENIED_CLAIMS:
        if claim["id"] == patient_id:
            # Return enriched mock data
            return {
                **claim,
                "gender": "Unknown",
                "dob": "",
                "phone": "",
                "email": "",
                "address": "",
                "denialDetails": f"Claim denied due to: {claim['denialReason']}",
                "requiredItems": ["Medical records", "Prior authorization documentation", "Physician notes"],
                "diagnosisCode": claim.get("procedure", "").split(" - ")[-1] if " - " in claim.get("procedure", "") else "",
                "medicalHistory": "",
                "medications": "",
                "allergies": "",
                "priorAuth": "",
            }
    
    # Fallback to claim_status.json if not in mock data
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
@app.get("/appeals/api/metrics/summary")
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

@app.get("/api/claim-journey/{patient_id}")
@app.get("/appeals/api/claim-journey/{patient_id}")
async def get_claim_journey(patient_id: str):
    """Return the full claim lifecycle for a patient — generated by pre-submission pipeline."""
    statuses = _load_claim_statuses()
    patients_csv = _load_patients_csv()

    entry = statuses.get(patient_id)
    if not entry:
        raise HTTPException(status_code=404, detail="No claim found for this patient")

    csv_row = patients_csv.get(patient_id, {})
    submission = entry.get("submission_result") or {}
    denial_info = submission.get("denial_info") or {}

    # Build timeline events from what the pre-submission pipeline recorded
    events = []

    # Event 1: Claim generated & submitted
    events.append({
        "type": "submitted",
        "title": "Claim Generated & Submitted",
        "description": f"Pre-submission AI pipeline processed patient data and submitted claim to {csv_row.get('insurer', 'insurer')}.",
        "timestamp": entry.get("timestamp", ""),
        "badge": "pending",
        "icon": "fa-paper-plane",
    })

    # Event 2: Insurer response
    sub_status = (submission.get("status") or "").lower()
    if sub_status == "approved":
        approved_amt = submission.get("approved_amount", 0)
        dq = submission.get("data_quality_score", 0)
        events.append({
            "type": "approved",
            "title": "Claim Approved",
            "description": f"Insurer approved the claim. Approved amount: ${approved_amt:.2f}. Data quality score: {dq}%.",
            "timestamp": entry.get("updated_at", entry.get("timestamp", "")),
            "badge": "approved",
            "icon": "fa-check-circle",
        })
    elif sub_status in ("rejected", "error"):
        reason = denial_info.get("reason") or submission.get("reason") or submission.get("message", "Claim denied by insurer")
        details = denial_info.get("details", "")
        required = denial_info.get("required_items", [])
        dq = submission.get("data_quality_score", 0)
        events.append({
            "type": "denied",
            "title": "Claim Denied by Insurer",
            "description": reason + (f" — {details}" if details else ""),
            "timestamp": entry.get("updated_at", entry.get("timestamp", "")),
            "badge": "denied",
            "icon": "fa-times-circle",
            "required_items": required,
            "data_quality_score": dq,
        })

    # Event 3: Appeal / resubmission status
    claim_status = (entry.get("status") or "").lower()
    if "appeal" in claim_status or "resubmit" in claim_status:
        if "resubmit" in claim_status:
            events.append({
                "type": "appeal",
                "title": "Appeal Generated & Resubmitted",
                "description": "AI appeal agent generated a corrected claim and resubmitted to the insurer.",
                "timestamp": entry.get("updated_at", ""),
                "badge": "resubmitted",
                "icon": "fa-redo",
            })
        elif "generated" in claim_status:
            events.append({
                "type": "appeal",
                "title": "Appeal Generated — Awaiting Submission",
                "description": "AI appeal agent has prepared the appeal. Pending manual review or auto-submission.",
                "timestamp": entry.get("updated_at", ""),
                "badge": "pending",
                "icon": "fa-file-signature",
            })

    return {
        "patient_id": patient_id,
        "claim_id": entry.get("claim_id", ""),
        "patient_name": csv_row.get("name", f"Patient {patient_id}"),
        "insurer": csv_row.get("insurer", "Unknown"),
        "procedure_code": csv_row.get("procedure_code", ""),
        "diagnosis_code": csv_row.get("diagnosis_code", ""),
        "claim_amount": float(csv_row.get("claim_amount", 0) or 0),
        "service_date": csv_row.get("service_date", ""),
        "provider": csv_row.get("provider", ""),
        "risk_score": entry.get("risk_score", 0),
        "issues_count": entry.get("issues_count", 0),
        "data_quality_score": submission.get("data_quality_score", 0),
        "processing_time_seconds": entry.get("processing_time", 0) if entry.get("processing_time", 0) < 300 else None,
        "current_status": entry.get("status", "unknown"),
        "events": events,
    }


@app.get("/api/pipeline-status")
@app.get("/appeals/api/pipeline-status")
async def get_pipeline_status():
    """Return the full agentic pipeline status — reads from claim_status.json and MCP logs."""
    statuses = _load_claim_statuses()
    patients_csv = _load_patients_csv()

    # Read MCP agent logs for recent activity
    logs_dir = os.path.join(PRE_SUBMISSION_DIR, "data", "logs")
    agent_logs = []
    mcp_files = [
        ("RiskPredictor-MCP_log.jsonl", "Risk Predictor"),
        ("AutoCorrector-MCP_log.jsonl", "Auto Corrector"),
        ("ClaimSubmitter-MCP_log.jsonl", "Claim Submitter"),
        ("AppealGenerator_log.jsonl", "Appeal Generator"),
        ("resubmitter_log.jsonl", "Resubmitter"),
        ("feedback_learner_log.jsonl", "Feedback Learner"),
    ]
    for fname, label in mcp_files:
        fpath = os.path.join(logs_dir, fname)
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            for line in lines[-5:]:
                try:
                    entry = json.loads(line)
                    state = entry.get("state_snapshot", {})
                    agent_logs.append({
                        "agent": label,
                        "timestamp": entry.get("timestamp", ""),
                        "claim_id": state.get("claim_id", entry.get("claim_id", "")),
                        "patient_id": state.get("patient_id", ""),
                        "risk_score": state.get("risk_score"),
                        "final_status": state.get("final_status", ""),
                        "action": entry.get("action", ""),
                    })
                except Exception:
                    pass
        except Exception:
            pass

    # Summary stats
    total = len(statuses)
    approved = sum(1 for e in statuses.values() if (e.get("status") or "").lower() == "approved")
    denied = sum(1 for e in statuses.values() if (e.get("status") or "").lower() in ("denied", "rejected"))
    resubmitted = sum(1 for e in statuses.values() if "resubmit" in (e.get("status") or "").lower())
    appeal_gen = sum(1 for e in statuses.values() if "appeal" in (e.get("status") or "").lower())

    return {
        "pipeline_summary": {
            "total_claims": total,
            "approved": approved,
            "denied": denied,
            "resubmitted": resubmitted,
            "appeal_generated": appeal_gen,
            "bedrock_available": BEDROCK_AVAILABLE,
            "model": "us.amazon.nova-micro-v1:0" if BEDROCK_AVAILABLE else "unavailable",
        },
        "recent_agent_activity": sorted(agent_logs, key=lambda x: x.get("timestamp", ""), reverse=True)[:20],
        "claim_statuses": {
            pid: {
                "status": e.get("status"),
                "claim_id": e.get("claim_id"),
                "risk_score": e.get("risk_score"),
                "timestamp": e.get("timestamp"),
                "patient_name": patients_csv.get(pid, {}).get("name", pid),
            }
            for pid, e in statuses.items()
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/bedrock-appeal/{patient_id}")
@app.post("/appeals/api/bedrock-appeal/{patient_id}")
async def generate_bedrock_appeal(patient_id: str):
    """Generate a full appeal letter using AWS Bedrock Nova Micro."""
    statuses = _load_claim_statuses()
    patients_csv = _load_patients_csv()

    entry = statuses.get(patient_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Patient not found")

    csv_row = patients_csv.get(patient_id, {})
    sub = entry.get("submission_result") or {}
    denial_info = sub.get("denial_info") or {}
    denial_reason = denial_info.get("reason") or sub.get("message", "Claim denied")
    code_info = _map_denial_code(denial_reason)

    prompt = f"""You are a medical billing specialist writing a formal insurance appeal letter.

Patient: {csv_row.get('name', patient_id)}
Insurer: {csv_row.get('insurer', 'Unknown')}
Claim ID: {entry.get('claim_id', '')}
Service Date: {csv_row.get('service_date', '')}
Procedure Code: {csv_row.get('procedure_code', '')} 
Diagnosis Code: {csv_row.get('diagnosis_code', '')}
Claim Amount: ${csv_row.get('claim_amount', 0)}
Denial Reason: {denial_reason}
Denial Code: {code_info['code']}
Provider: {csv_row.get('provider', '')}
Medical History: {csv_row.get('medical_history', '')}
Prior Authorization: {csv_row.get('prior_authorization', 'N/A')}

Write a professional, concise appeal letter (3-4 paragraphs) that:
1. States the appeal purpose and claim details
2. Addresses the specific denial reason with clinical justification
3. Cites medical necessity and supporting evidence
4. Requests reconsideration with a clear call to action

Use formal medical billing language. Address it to the Medical Director."""

    appeal_text = _bedrock_invoke(prompt, max_tokens=800)

    if not appeal_text:
        # Fallback template
        appeal_text = f"""Dear Medical Director,

We are writing to formally appeal the denial of claim {entry.get('claim_id', '')} for patient {csv_row.get('name', patient_id)} (ID: {patient_id}).

The claim was denied with reason: {denial_reason} (Code: {code_info['code']}). We respectfully disagree with this determination. The procedure ({csv_row.get('procedure_code', '')}) was medically necessary for the treatment of {csv_row.get('diagnosis_code', '')} as documented in the patient's medical records.

We request a full reconsideration of this claim. All supporting documentation including clinical notes, prior authorization records, and medical necessity documentation is available upon request.

Sincerely,
{csv_row.get('provider', 'Healthcare Provider')}"""

    return {
        "patient_id": patient_id,
        "claim_id": entry.get("claim_id", ""),
        "patient_name": csv_row.get("name", patient_id),
        "denial_code": code_info["code"],
        "denial_reason": denial_reason,
        "appeal_letter": appeal_text,
        "generated_by": "AWS Bedrock Nova Micro" if (BEDROCK_AVAILABLE and appeal_text) else "Template Fallback",
        "generated_at": datetime.now().isoformat(),
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
@app.post("/appeals/api/process-era/{patient_id}")
async def process_era_for_patient(patient_id: str, request: Request):
    """Process an uploaded ERA/835 file using AI agents + AWS Bedrock Nova Micro.

    Pipeline:
    1. Parse X12 835 segments (CLP, CAS, NM1, SVC, DTM)
    2. Classify each denial code using rule-based + Bedrock AI
    3. Generate appeal strategy via Bedrock Nova Micro
    4. Return structured analysis for the frontend
    """
    body = await request.json()
    era_text = body.get("era_content", "")
    filename = body.get("filename", "uploaded.835")

    if not era_text.strip():
        raise HTTPException(status_code=400, detail="ERA content is empty")

    # ── Step 1: Parse ERA/835 ──────────────────────────────────────────────
    if era_processor:
        era_result = era_processor.process_era_file(era_text, filename)
    else:
        era_result = _fallback_parse_era(era_text, filename)

    # ── Step 2: Classify denials ──────────────────────────────────────────
    classified = []
    if denial_classifier and era_result.get("denials_extracted"):
        classified = denial_classifier.batch_classify_denials(era_result["denials_extracted"])
    elif era_result.get("denials_extracted"):
        classified = era_result["denials_extracted"]

    # ── Step 3: Bedrock AI analysis ───────────────────────────────────────
    llm_analysis = None
    denials_for_ai = (classified or era_result.get("denials_extracted", []))[:5]
    if denials_for_ai:
        denial_summary = "\n".join([
            f"- Claim {d.get('claim_id','?')}: {d.get('denial_code','?')} — {d.get('denial_reason','?')} (${d.get('denied_amount',0):.2f})"
            for d in denials_for_ai
        ])
        # Load patient context
        statuses = _load_claim_statuses()
        patients_csv = _load_patients_csv()
        entry = statuses.get(patient_id, {})
        csv_row = patients_csv.get(patient_id, {})
        patient_context = (
            f"Patient: {csv_row.get('name','Unknown')}, Insurer: {csv_row.get('insurer','Unknown')}, "
            f"Procedure: {csv_row.get('procedure_code','?')}, Diagnosis: {csv_row.get('diagnosis_code','?')}, "
            f"Amount: ${csv_row.get('claim_amount',0)}, Prior Auth: {csv_row.get('prior_authorization','N/A')}"
        )
        prompt = f"""You are a healthcare claims expert analyzing an Electronic Remittance Advice (ERA/835).

Patient Context: {patient_context}

Denied Claims in this ERA:
{denial_summary}

Provide a concise JSON analysis with these exact keys:
{{
  "overall_assessment": "2-sentence summary of the denial pattern",
  "root_cause_analysis": "primary root cause in 1 sentence",
  "risk_level": "high|medium|low",
  "top_recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"],
  "appeal_priority_order": ["denial code 1", "denial code 2"],
  "estimated_recovery": "dollar amount or percentage estimate",
  "process_improvement": "1 actionable process improvement"
}}

Respond with ONLY the JSON object, no markdown."""

        ai_text = _bedrock_invoke(prompt, max_tokens=600)
        if ai_text:
            try:
                # Strip markdown code fences if present
                clean = ai_text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                llm_analysis = json.loads(clean)
                llm_analysis["powered_by"] = "AWS Bedrock — Nova Micro"
            except Exception:
                llm_analysis = {
                    "overall_assessment": ai_text[:300],
                    "powered_by": "AWS Bedrock — Nova Micro",
                    "error": None
                }

    # ── Step 4: Appeal suggestions ────────────────────────────────────────
    appeal_suggestions = []
    for denial in denials_for_ai[:3]:
        denial_data = {
            "claim_id": denial.get("claim_id", ""),
            "denial_code": denial.get("denial_code", ""),
            "denial_reason": denial.get("denial_reason", ""),
            "denied_amount": denial.get("denied_amount", 0),
            "payer": denial.get("payer", "Unknown"),
        }
        if appeal_generator:
            appeal = appeal_generator.generate_appeal(denial_data, denial.get("classification"))
            appeal_suggestions.append({
                "claim_id": denial_data["claim_id"],
                "denial_code": denial_data["denial_code"],
                "strategy": appeal.get("appeal_strategy", ""),
                "success_probability": appeal.get("success_probability", 0),
                "estimated_time": appeal.get("estimated_completion_time", ""),
            })

    # ── Step 5: Statistics ────────────────────────────────────────────────
    stats = {}
    if era_processor:
        stats = era_processor.get_era_statistics(era_result)

    return {
        "status": "processed",
        "filename": filename,
        "summary": era_result.get("summary", {}),
        "denials_count": len(era_result.get("denials_extracted", [])),
        "classified_denials": classified[:5],
        "appeal_suggestions": appeal_suggestions,
        "statistics": stats,
        "llm_analysis": llm_analysis,
        "bedrock_available": BEDROCK_AVAILABLE,
        "processed_at": datetime.now().isoformat(),
    }
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

@app.post("/api/resubmit-corrected-claim/{patient_id}")
@app.post("/appeals/api/resubmit-corrected-claim/{patient_id}")
async def resubmit_corrected_claim(patient_id: str, request: Request):
    """Generate a corrected X12 837P claim incorporating ERA denial findings.

    Uses Bedrock Nova Micro to:
    1. Analyse the denial reason and ERA findings
    2. Generate specific field corrections
    3. Build a corrected X12 837P claim transaction
    4. Return the full claim with a plain-language correction summary
    """
    body = await request.json()
    denial_code    = body.get("denial_code", "CO-16")
    denial_reason  = body.get("denial_reason", "")
    era_assessment = body.get("era_assessment", "")
    recommendations = body.get("recommendations", [])

    statuses    = _load_claim_statuses()
    patients_csv = _load_patients_csv()

    entry   = statuses.get(patient_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Patient not found")

    csv_row = patients_csv.get(patient_id, {})
    sub     = entry.get("submission_result") or {}
    denial_info = sub.get("denial_info") or {}

    # ── Patient & claim data ──────────────────────────────────────────────
    patient_name   = csv_row.get("name", patient_id)
    insurer        = csv_row.get("insurer", "Unknown")
    procedure_code = csv_row.get("procedure_code", "99213")
    diagnosis_code = csv_row.get("diagnosis_code", "Z00.00")
    claim_amount   = float(csv_row.get("claim_amount", 0) or 0)
    service_date   = csv_row.get("service_date", datetime.now().strftime("%Y-%m-%d"))
    provider       = csv_row.get("provider", "Healthcare Provider")
    prior_auth     = csv_row.get("prior_authorization", "N/A")
    medical_history = csv_row.get("medical_history", "")
    dob            = csv_row.get("dob", "19800101")
    gender         = csv_row.get("gender", "U")
    original_claim_id = entry.get("claim_id", f"CLM-{patient_id}")

    # ── Step 1: Bedrock — generate corrections ────────────────────────────
    recs_text = "\n".join([f"- {r}" for r in (recommendations or [denial_reason])[:5]])
    correction_prompt = f"""You are a medical billing expert correcting a denied insurance claim.

ORIGINAL CLAIM:
- Patient: {patient_name} (ID: {patient_id})
- Insurer: {insurer}
- Original Claim ID: {original_claim_id}
- Service Date: {service_date}
- Procedure Code: {procedure_code}
- Diagnosis Code: {diagnosis_code}
- Claim Amount: ${claim_amount}
- Provider: {provider}
- Prior Authorization: {prior_auth}
- Medical History: {medical_history}

DENIAL INFORMATION:
- Denial Code: {denial_code}
- Denial Reason: {denial_reason}
- ERA Assessment: {era_assessment}

RECOMMENDED CORRECTIONS:
{recs_text}

Generate a JSON object with these exact keys describing the corrected claim:
{{
  "correction_summary": "2-3 sentence plain English explanation of what was wrong and what was corrected",
  "corrections_made": ["specific correction 1", "specific correction 2", "specific correction 3"],
  "corrected_procedure_code": "{procedure_code}",
  "corrected_diagnosis_code": "{diagnosis_code}",
  "corrected_modifier": "modifier if needed, else empty string",
  "corrected_prior_auth": "corrected prior auth note",
  "additional_documentation": ["document 1 to attach", "document 2 to attach"],
  "appeal_strength": "strong|moderate|weak",
  "estimated_approval_probability": "percentage as string e.g. 78%",
  "resubmission_notes": "1 sentence note for the billing team"
}}

Respond with ONLY the JSON object."""

    corrections_json = {}
    ai_text = _bedrock_invoke(correction_prompt, max_tokens=700)
    if ai_text:
        try:
            clean = ai_text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            corrections_json = json.loads(clean)
        except Exception:
            corrections_json = {
                "correction_summary": ai_text[:400],
                "corrections_made": [denial_reason],
                "appeal_strength": "moderate",
                "estimated_approval_probability": "65%",
            }

    # Apply corrections to claim fields
    corr_proc  = corrections_json.get("corrected_procedure_code") or procedure_code
    corr_diag  = corrections_json.get("corrected_diagnosis_code") or diagnosis_code
    corr_mod   = corrections_json.get("corrected_modifier", "")
    corr_auth  = corrections_json.get("corrected_prior_auth") or prior_auth

    # ── Step 2: Build corrected X12 837P ─────────────────────────────────
    now        = datetime.now()
    date_str   = now.strftime("%Y%m%d")
    time_str   = now.strftime("%H%M")
    new_claim_id = f"CLM-{patient_id}-CORR-{now.strftime('%Y%m%d%H%M%S')}"

    # Format service date for X12 (YYYYMMDD)
    try:
        svc_date_x12 = datetime.strptime(service_date, "%Y-%m-%d").strftime("%Y%m%d")
    except Exception:
        svc_date_x12 = date_str

    # Format DOB for X12
    try:
        dob_x12 = datetime.strptime(str(dob), "%Y-%m-%d").strftime("%Y%m%d")
    except Exception:
        dob_x12 = str(dob).replace("-", "")[:8] or "19800101"

    gender_x12 = "M" if str(gender).upper().startswith("M") else "F" if str(gender).upper().startswith("F") else "U"
    proc_with_mod = corr_proc + (f":{corr_mod}" if corr_mod else "")

    x12_837p = (
        f"ISA*00*          *00*          *ZZ*MEDICLAIMS      *ZZ*{insurer[:15]:<15}*"
        f"{date_str[2:]}*{time_str}*^*00501*000000001*0*P*:~\n"
        f"GS*HC*MEDICLAIMS*{insurer[:15]}*{date_str}*{time_str}*1*X*005010X222A1~\n"
        f"ST*837*0001*005010X222A1~\n"
        f"BPR*22*{claim_amount:.2f}*C*ACH*CCP*01*999999999*DA*123456789*1234567890**01*"
        f"987654321*DA*987654321*{date_str}~\n"
        f"NM1*41*2*MEDICLAIMS AI*****46*MEDICLAIMS01~\n"
        f"PER*IC*BILLING DEPT*TE*5551234567~\n"
        f"NM1*40*2*{insurer.upper()[:35]}*****46*{insurer[:10].upper()}01~\n"
        f"HL*1**20*1~\n"
        f"PRV*BI*PXC*207Q00000X~\n"
        f"NM1*85*2*{provider.upper()[:35]}*****XX*1234567890~\n"
        f"N3*123 MEDICAL CENTER DR~\n"
        f"N4*ANYTOWN*NY*10001~\n"
        f"REF*EI*123456789~\n"
        f"HL*2*1*22*0~\n"
        f"SBR*P*18*******MC~\n"
        f"NM1*IL*1*{patient_name.split()[-1].upper() if patient_name else 'PATIENT'}*"
        f"{patient_name.split()[0].upper() if patient_name else 'UNKNOWN'}****MI*{patient_id}~\n"
        f"N3*456 PATIENT ST~\n"
        f"N4*ANYTOWN*NY*10001~\n"
        f"DMG*D8*{dob_x12}*{gender_x12}~\n"
        f"NM1*PR*2*{insurer.upper()[:35]}*****PI*{insurer[:8].upper()}001~\n"
        f"CLM*{new_claim_id}*{claim_amount:.2f}***11:B:1*Y*A*Y*I~\n"
        f"DTP*434*D8*{svc_date_x12}~\n"
        f"REF*D9*{original_claim_id}~\n"
        f"REF*9F*{corr_auth}~\n"
        f"NTE*ADD*CORRECTED RESUBMISSION - ORIGINAL DENIED: {denial_code}~\n"
        f"NTE*ADD*CORRECTION: {(corrections_json.get('corrections_made') or [denial_reason])[0][:60]}~\n"
        f"HI*ABK:{corr_diag.replace('.', '')}~\n"
        f"NM1*82*1*{provider.split()[-1].upper() if provider else 'PROVIDER'}*"
        f"{provider.split()[0].upper() if provider else 'DR'}****XX*1234567890~\n"
        f"LX*1~\n"
        f"SV1*HC:{proc_with_mod}*{claim_amount:.2f}*UN*1***1~\n"
        f"DTP*472*D8*{svc_date_x12}~\n"
        f"SE*28*0001~\n"
        f"GE*1*1~\n"
        f"IEA*1*000000001~"
    )

    # ── Step 3: Build human-readable claim summary ────────────────────────
    corrections_made = corrections_json.get("corrections_made") or [f"Addressed {denial_code} denial"]
    add_docs = corrections_json.get("additional_documentation") or []

    return {
        "status": "corrected_claim_generated",
        "new_claim_id": new_claim_id,
        "original_claim_id": original_claim_id,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "insurer": insurer,
        "claim_amount": claim_amount,
        "service_date": service_date,
        "procedure_code": corr_proc,
        "diagnosis_code": corr_diag,
        "modifier": corr_mod,
        "prior_auth": corr_auth,
        "provider": provider,
        "denial_code_addressed": denial_code,
        "correction_summary": corrections_json.get("correction_summary", f"Claim corrected to address {denial_code} denial."),
        "corrections_made": corrections_made,
        "additional_documentation": add_docs,
        "appeal_strength": corrections_json.get("appeal_strength", "moderate"),
        "estimated_approval_probability": corrections_json.get("estimated_approval_probability", "70%"),
        "resubmission_notes": corrections_json.get("resubmission_notes", ""),
        "x12_837p": x12_837p,
        "generated_by": "AWS Bedrock Nova Micro" if BEDROCK_AVAILABLE else "Template",
        "generated_at": now.isoformat(),
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
