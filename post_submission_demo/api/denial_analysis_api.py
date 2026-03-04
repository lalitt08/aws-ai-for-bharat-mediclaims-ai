"""
Denial Analysis API endpoints
Uses REAL claim data from claim_status.json + patients CSV
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import os, json, csv

denial_router = APIRouter()

# ── Shared data paths ──
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PRE_DIR = os.path.dirname(_BASE)
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

# Denial code knowledge base
DENIAL_CODES = {
    "CO-4": "The procedure code is inconsistent with the modifier used or a required modifier is missing",
    "CO-16": "Claim/service lacks information needed for adjudication",
    "CO-18": "Exact duplicate claim/service",
    "CO-27": "Expenses incurred after coverage terminated",
    "CO-29": "The time limit for filing has expired",
    "CO-50": "These are non-covered services because this is not deemed a medical necessity",
    "CO-96": "Non-covered charge(s)",
    "CO-97": "The benefit for this service is included in the payment for another service",
    "CO-109": "Claim not covered by this payer/contractor",
    "CO-151": "Information does not support this many/frequency of services",
    "CO-197": "Precertification/authorization/notification absent",
    "CO-204": "This service/equipment/drug is not covered under the patient's current benefit plan",
}

CATEGORY_MAP = {
    "CO-4": "coding_error", "CO-16": "documentation", "CO-18": "duplicate_claim",
    "CO-27": "eligibility", "CO-29": "timely_filing", "CO-50": "medical_necessity",
    "CO-96": "medical_necessity", "CO-97": "bundling", "CO-109": "eligibility",
    "CO-151": "medical_necessity", "CO-197": "prior_authorization", "CO-204": "policy_exclusion",
}

def _map_code(reason: str) -> str:
    r = (reason or "").lower()
    mapping = {
        "missing clinical documentation": "CO-16", "incomplete provider credentials": "CO-16",
        "diagnosis code mismatch": "CO-4", "modifier usage error": "CO-4",
        "prior authorization expired": "CO-197", "service level mismatch": "CO-197",
        "medical necessity not established": "CO-50", "timely filing": "CO-29",
        "duplicate claim": "CO-18",
    }
    for key, code in mapping.items():
        if key in r:
            return code
    return "CO-16"


@denial_router.get("/analyze/{claim_id}")
async def analyze_denial(claim_id: str):
    """Analyze denial for a specific claim using real data."""
    statuses = _load_statuses()
    patients = _load_patients()

    # Find by claim_id or patient_id
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
    reason = denial_info.get("reason") or sub.get("message", "Unknown")
    code = _map_code(reason)
    category = CATEGORY_MAP.get(code, "other")
    details = denial_info.get("details", "")
    required = denial_info.get("required_items", [])
    success_rate = denial_info.get("success_rate", 0.5)

    # Build actionable items from required_items
    actionable = required if required else [
        "Review medical records for supporting documentation",
        "Contact physician for additional clinical notes",
        "Verify procedure code accuracy",
    ]

    # Suggested fixes based on category
    fix_map = {
        "documentation": ["Submit missing clinical documentation", "Provide complete medical records", "Include provider attestation"],
        "coding_error": ["Correct procedure/diagnosis codes", "Add required modifier", "Verify code linkage"],
        "prior_authorization": ["Obtain retroactive prior authorization", "Document emergency circumstances", "Submit clinical justification"],
        "medical_necessity": ["Provide medical necessity justification", "Submit clinical guidelines", "Request peer-to-peer review"],
        "eligibility": ["Verify patient eligibility dates", "Check coverage status", "Appeal with eligibility documentation"],
        "timely_filing": ["Document good cause for late filing", "Provide proof of original submission", "Appeal with explanation"],
    }
    fixes = fix_map.get(category, ["Review denial and gather evidence", "Submit formal appeal"])

    return {
        "claim_id": entry.get("claim_id", claim_id),
        "patient_id": pid,
        "patient_name": csv_row.get("name", f"Patient {pid}"),
        "denial_codes": [code],
        "primary_reason": reason,
        "secondary_reasons": [],
        "details": details,
        "severity_score": round(entry.get("risk_score", 0.5), 2),
        "actionable_items": actionable,
        "suggested_fixes": fixes,
        "compliance_issues": [],
        "appeal_probability": round(float(success_rate), 2),
        "category": category,
        "payer": csv_row.get("insurer", "Unknown"),
        "amount": float(csv_row.get("claim_amount", 0) or 0),
    }


@denial_router.get("/denial-categories")
async def get_denial_categories():
    """Get denial categories with real counts from claim data."""
    statuses = _load_statuses()

    cat_counts = {}
    for entry in statuses.values():
        sub = entry.get("submission_result") or {}
        denial_info = sub.get("denial_info") or {}
        reason = denial_info.get("reason", "")
        if not reason:
            continue
        code = _map_code(reason)
        cat = CATEGORY_MAP.get(code, "other")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    categories = {
        "medical_necessity": {"name": "Medical Necessity", "description": "Service not deemed medically necessary", "common_codes": ["CO-50", "CO-96", "CO-197"], "appeal_strategy": "Provide clinical documentation supporting medical necessity"},
        "prior_authorization": {"name": "Prior Authorization", "description": "Prior authorization required but not obtained", "common_codes": ["CO-27", "CO-197"], "appeal_strategy": "Obtain retroactive authorization or demonstrate emergency nature"},
        "documentation": {"name": "Insufficient Documentation", "description": "Inadequate supporting documentation provided", "common_codes": ["CO-16", "CO-29"], "appeal_strategy": "Submit complete medical records and clinical notes"},
        "coding_error": {"name": "Coding Error", "description": "Incorrect procedure or diagnosis codes", "common_codes": ["CO-4", "CO-109"], "appeal_strategy": "Correct codes and resubmit with proper documentation"},
        "timely_filing": {"name": "Timely Filing", "description": "Claim submitted after filing deadline", "common_codes": ["CO-29", "CO-204"], "appeal_strategy": "Demonstrate good cause for late filing"},
        "policy_exclusion": {"name": "Policy Exclusion", "description": "Service excluded under patient's policy", "common_codes": ["CO-97", "CO-109"], "appeal_strategy": "Challenge exclusion interpretation or demonstrate coverage"},
    }

    for key in categories:
        categories[key]["count"] = cat_counts.get(key, 0)

    return categories


@denial_router.get("/denial-trends")
async def get_denial_trends():
    """Get denial trends computed from real claim data."""
    statuses = _load_statuses()
    patients = _load_patients()

    reason_counts = {}
    payer_stats = {}

    for pid, entry in statuses.items():
        sub = entry.get("submission_result") or {}
        denial_info = sub.get("denial_info") or {}
        reason = denial_info.get("reason", "")
        csv_row = patients.get(pid, {})
        payer = csv_row.get("insurer", "Unknown")
        status = (entry.get("status") or "").lower()

        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        if payer not in payer_stats:
            payer_stats[payer] = {"total": 0, "denied": 0, "appealed": 0}
        payer_stats[payer]["total"] += 1
        if status not in ("approved",):
            payer_stats[payer]["denied"] += 1
        if "appeal" in status or "resubmit" in status:
            payer_stats[payer]["appealed"] += 1

    total_reasons = sum(reason_counts.values()) or 1
    sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)

    payer_perf = []
    for payer, stats in payer_stats.items():
        denial_rate = stats["denied"] / stats["total"] if stats["total"] else 0
        appeal_success = stats["appealed"] / stats["denied"] if stats["denied"] else 0
        payer_perf.append({
            "payer": payer,
            "denial_rate": round(denial_rate, 2),
            "appeal_success": round(min(appeal_success, 1.0), 2),
            "total_claims": stats["total"],
        })

    return {
        "top_denial_reasons": [
            {"reason": r, "count": c, "percentage": round(c / total_reasons * 100, 1)}
            for r, c in sorted_reasons
        ],
        "payer_performance": payer_perf,
    }
