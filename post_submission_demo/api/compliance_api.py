"""
Compliance API endpoints - Uses real claim data + ComplianceChecker service
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import os, json, csv

compliance_router = APIRouter()

_PRE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLAIM_FILE = os.path.join(_PRE_DIR, "data", "claim_status.json")
_CSV_FILE = os.path.join(_PRE_DIR, "data", "patients1.csv")

def _load_statuses():
    try:
        if os.path.exists(_CLAIM_FILE):
            with open(_CLAIM_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _load_patients():
    patients = {}
    try:
        if os.path.exists(_CSV_FILE):
            with open(_CSV_FILE, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    pid = row.get("patient_id", "").strip()
                    if pid:
                        patients[pid] = row
    except Exception:
        pass
    return patients

# Try to import the real ComplianceChecker service
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from services.compliance_checker import ComplianceChecker
    checker = ComplianceChecker()
except ImportError:
    checker = None


@compliance_router.get("/check/{claim_id}")
async def check_compliance(claim_id: str):
    """Compliance check using real claim data + ComplianceChecker service."""
    statuses = _load_statuses()
    patients = _load_patients()

    # Find entry
    entry = None
    pid = None
    for p, e in statuses.items():
        if e.get("claim_id") == claim_id or p == claim_id:
            entry = e
            pid = p
            break

    if not entry:
        raise HTTPException(404, "Claim not found")

    csv_row = patients.get(pid, {})
    sub = entry.get("submission_result") or {}
    denial_info = sub.get("denial_info") or {}

    claim_data = {
        "claim_id": entry.get("claim_id", claim_id),
        "patient_name": csv_row.get("name", ""),
        "service_date": csv_row.get("service_date", ""),
        "payer": (csv_row.get("insurer") or "").lower(),
        "claim_amount": float(csv_row.get("claim_amount", 0) or 0),
        "procedure_code": csv_row.get("procedure_code", ""),
        "prior_authorization": csv_row.get("prior_authorization", ""),
    }

    if checker:
        result = checker.check_claim_compliance(claim_data)
        return result

    # Fallback: basic compliance assessment
    violations = []
    warnings = []
    recommendations = []
    score = 1.0

    quality = float(sub.get("data_quality_score", 50) or 50)
    risk = entry.get("risk_score", 0.5)

    if quality < 40:
        violations.append({
            "type": "DOCUMENTATION",
            "description": f"Data quality score is very low ({quality}%). Documentation likely incomplete.",
            "severity": "high",
            "required_action": "Review and complete all required documentation fields",
        })
        score -= 0.25

    if risk > 0.8:
        warnings.append({
            "type": "HIGH_RISK",
            "description": f"Claim has high risk score ({risk:.2f}). Multiple issues detected.",
            "severity": "medium",
            "required_action": "Address all identified issues before resubmission",
        })
        score -= 0.1

    if not claim_data.get("prior_authorization"):
        warnings.append({
            "type": "PRIOR_AUTH",
            "description": "No prior authorization on file",
            "severity": "medium",
            "required_action": "Verify if prior authorization is required for this service",
        })
        score -= 0.05

    required = denial_info.get("required_items", [])
    if required:
        recommendations.extend([f"Provide: {item}" for item in required])

    status = "violation" if violations else ("warning" if warnings else "compliant")

    return {
        "claim_id": entry.get("claim_id", claim_id),
        "compliance_status": status,
        "compliance_score": round(max(0.0, score), 2),
        "violations": violations,
        "warnings": warnings,
        "recommendations": recommendations or ["No immediate compliance issues identified"],
        "checked_at": datetime.now().isoformat(),
        "data_quality_score": quality,
        "risk_score": risk,
    }


@compliance_router.get("/rules")
async def get_compliance_rules():
    """Get compliance rules."""
    return [
        {"rule_id": "RULE-001", "category": "HIPAA", "description": "Patient authorization required for disclosure of PHI", "severity": "high", "regulation_source": "45 CFR § 164.508"},
        {"rule_id": "RULE-002", "category": "Timely Filing", "description": "Claims must be filed within regulatory timeframe", "severity": "high", "regulation_source": "State Insurance Code"},
        {"rule_id": "RULE-003", "category": "Prior Authorization", "description": "High-cost procedures require prior authorization", "severity": "medium", "regulation_source": "Payer Policy Manual"},
        {"rule_id": "RULE-004", "category": "Documentation", "description": "Medical records must support billed services", "severity": "medium", "regulation_source": "CMS Guidelines"},
        {"rule_id": "RULE-005", "category": "Coding Accuracy", "description": "Procedure and diagnosis codes must be consistent", "severity": "medium", "regulation_source": "AMA CPT Guidelines"},
    ]


@compliance_router.get("/payer-policies/{payer}")
async def get_payer_policies(payer: str):
    """Get payer-specific compliance policies."""
    if checker:
        return checker.payer_policies.get(payer.lower(), {})

    policies = {
        "aetna": {
            "prior_authorization": {"threshold_amount": 1000, "required_procedures": ["MRI", "CT Scan", "Surgery"]},
            "appeal_process": {"deadline_days": 60, "required_documents": ["Original claim", "Medical records", "Physician letter"]},
        },
        "united": {
            "prior_authorization": {"threshold_amount": 750, "required_procedures": ["MRI", "CT Scan", "Surgery", "PT"]},
            "appeal_process": {"deadline_days": 45, "required_documents": ["Formal appeal letter", "Complete medical records", "Clinical guidelines"]},
        },
        "bluecross": {
            "prior_authorization": {"threshold_amount": 500, "required_procedures": ["Surgery", "DME", "Specialty drugs"]},
            "appeal_process": {"deadline_days": 90, "required_documents": ["Provider letter", "Medical necessity documentation", "Treatment history"]},
        },
        "cigna": {
            "prior_authorization": {"threshold_amount": 800, "required_procedures": ["MRI", "Surgery", "Specialty drugs"]},
            "appeal_process": {"deadline_days": 60, "required_documents": ["Appeal letter", "Clinical documentation", "Provider credentials"]},
        },
    }
    return policies.get(payer.lower(), {})


@compliance_router.post("/validate-appeal/{appeal_id}")
async def validate_appeal_compliance(appeal_id: str):
    """Validate compliance for an appeal."""
    statuses = _load_statuses()

    # Try to find by patient ID (appeal_id might be APP-PATxxx format)
    pid = appeal_id.replace("APP-", "")
    entry = statuses.get(pid)

    checks = []
    overall_score = 0.85

    if entry:
        sub = entry.get("submission_result") or {}
        quality = float(sub.get("data_quality_score", 50) or 50)

        checks.append({"check_type": "Data Quality", "status": "passed" if quality > 40 else "warning", "details": f"Data quality score: {quality}%"})
        checks.append({"check_type": "Claim Status", "status": "passed", "details": f"Current status: {entry.get('status', 'unknown')}"})
        checks.append({"check_type": "Risk Assessment", "status": "passed" if entry.get("risk_score", 0) < 0.8 else "warning", "details": f"Risk score: {entry.get('risk_score', 0):.2f}"})

        if quality < 40:
            overall_score -= 0.15
        if entry.get("risk_score", 0) > 0.8:
            overall_score -= 0.10
    else:
        checks.append({"check_type": "Claim Lookup", "status": "warning", "details": "Claim not found in system"})
        overall_score = 0.5

    return {
        "appeal_id": appeal_id,
        "validation_status": "compliant" if overall_score > 0.7 else "warning",
        "checks_performed": checks,
        "overall_score": round(overall_score, 2),
        "validated_at": datetime.now().isoformat(),
    }
