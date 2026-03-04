"""
Appeals API endpoints - Uses REAL claim data from claim_status.json
Handles appeal operations on actual denied claims
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import os, json, csv

appeals_router = APIRouter()

_PRE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLAIM_FILE = os.path.join(_PRE_DIR, "data", "claim_status.json")
_CSV_FILE = os.path.join(_PRE_DIR, "data", "patients1.csv")

# In-memory appeal actions log (persists during server lifetime)
_appeal_actions: Dict[str, Dict[str, Any]] = {}

def _load_statuses():
    try:
        if os.path.exists(_CLAIM_FILE):
            with open(_CLAIM_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_statuses(data):
    try:
        with open(_CLAIM_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving claim statuses: {e}")

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

CODE_MAP = {
    "missing clinical documentation": "CO-16", "incomplete provider credentials": "CO-16",
    "diagnosis code mismatch": "CO-4", "modifier usage error": "CO-4",
    "prior authorization expired": "CO-197", "service level mismatch": "CO-197",
    "medical necessity not established": "CO-50",
}

def _map_code(reason):
    r = (reason or "").lower()
    for key, code in CODE_MAP.items():
        if key in r:
            return code
    return "CO-16"

CATEGORY_MAP = {
    "CO-4": "coding_error", "CO-16": "documentation", "CO-50": "medical_necessity",
    "CO-197": "prior_authorization",
}

def _build_appeal(pid, entry, csv_row):
    sub = entry.get("submission_result") or {}
    denial_info = sub.get("denial_info") or {}
    reason = denial_info.get("reason") or sub.get("message", "Claim denied")
    code = _map_code(reason)
    amount = float(csv_row.get("claim_amount", 0) or 0)
    risk = entry.get("risk_score", 0.5)
    success_rate = denial_info.get("success_rate", 0.5)

    # Merge any in-memory actions
    actions = _appeal_actions.get(pid, {})

    return {
        "appeal_id": f"APP-{pid}",
        "claim_id": entry.get("claim_id", ""),
        "patient_id": pid,
        "patient_name": csv_row.get("name", f"Patient {pid}"),
        "payer": csv_row.get("insurer", "Unknown"),
        "original_amount": amount,
        "denial_reason": reason,
        "denial_code": code,
        "denial_category": CATEGORY_MAP.get(code, "documentation"),
        "status": actions.get("status", entry.get("status", "denied")),
        "priority": "high" if risk > 0.7 else ("medium" if risk > 0.4 else "low"),
        "created_date": entry.get("timestamp", ""),
        "last_updated": actions.get("last_updated", entry.get("updated_at", "")),
        "risk_score": risk,
        "success_probability": round(float(success_rate) * 100 if float(success_rate) <= 1 else float(success_rate)),
        "required_items": denial_info.get("required_items", []),
        "notes": actions.get("notes", ""),
    }


@appeals_router.get("/")
async def get_appeals(
    status: Optional[str] = Query(None),
    payer: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    limit: int = Query(20),
    offset: int = Query(0),
):
    """Get appeals from real denied claims."""
    statuses = _load_statuses()
    patients = _load_patients()

    appeals = []
    for pid, entry in statuses.items():
        s = (entry.get("status") or "").lower()
        if s == "approved":
            continue
        csv_row = patients.get(pid, {})
        appeals.append(_build_appeal(pid, entry, csv_row))

    # Apply filters
    if status:
        appeals = [a for a in appeals if a["status"] == status]
    if payer:
        appeals = [a for a in appeals if payer.lower() in a["payer"].lower()]
    if priority:
        appeals = [a for a in appeals if a["priority"] == priority]

    # Sort by risk score descending
    appeals.sort(key=lambda x: x.get("risk_score", 0), reverse=True)

    return appeals[offset:offset + limit]


@appeals_router.get("/stats/summary")
async def get_appeals_summary():
    """Summary stats from real data."""
    statuses = _load_statuses()
    patients = _load_patients()

    status_counts = {}
    priority_counts = {}
    payer_counts = {}

    for pid, entry in statuses.items():
        s = (entry.get("status") or "").lower()
        if s == "approved":
            continue
        csv_row = patients.get(pid, {})
        appeal = _build_appeal(pid, entry, csv_row)

        status_counts[appeal["status"]] = status_counts.get(appeal["status"], 0) + 1
        priority_counts[appeal["priority"]] = priority_counts.get(appeal["priority"], 0) + 1
        payer_counts[appeal["payer"]] = payer_counts.get(appeal["payer"], 0) + 1

    return {
        "total_appeals": sum(status_counts.values()),
        "status_distribution": status_counts,
        "priority_distribution": priority_counts,
        "payer_distribution": payer_counts,
        "last_updated": datetime.now().isoformat(),
    }


@appeals_router.get("/{appeal_id}")
async def get_appeal(appeal_id: str):
    """Get a single appeal."""
    pid = appeal_id.replace("APP-", "")
    statuses = _load_statuses()
    patients = _load_patients()

    entry = statuses.get(pid)
    if not entry:
        raise HTTPException(404, "Appeal not found")

    csv_row = patients.get(pid, {})
    return _build_appeal(pid, entry, csv_row)


class AppealUpdate(BaseModel):
    status: Optional[str] = None
    priority: Optional[str] = None
    notes: Optional[str] = None

@appeals_router.put("/{appeal_id}")
async def update_appeal(appeal_id: str, update: AppealUpdate):
    """Update an appeal - persists status changes to claim_status.json."""
    pid = appeal_id.replace("APP-", "")
    statuses = _load_statuses()

    if pid not in statuses:
        raise HTTPException(404, "Appeal not found")

    # Update in-memory actions
    if pid not in _appeal_actions:
        _appeal_actions[pid] = {}

    now = datetime.now().isoformat()
    if update.status:
        _appeal_actions[pid]["status"] = update.status
        statuses[pid]["status"] = update.status
        statuses[pid]["updated_at"] = now
    if update.notes:
        _appeal_actions[pid]["notes"] = update.notes
    _appeal_actions[pid]["last_updated"] = now

    # Persist to file
    _save_statuses(statuses)

    return {"message": "Appeal updated successfully", "appeal_id": appeal_id}


@appeals_router.post("/{appeal_id}/resubmit")
async def resubmit_appeal(appeal_id: str):
    """Resubmit an appeal - updates status in claim_status.json."""
    pid = appeal_id.replace("APP-", "")
    statuses = _load_statuses()

    if pid not in statuses:
        raise HTTPException(404, "Appeal not found")

    now = datetime.now().isoformat()
    statuses[pid]["status"] = "appeal_resubmitted"
    statuses[pid]["updated_at"] = now

    if pid not in _appeal_actions:
        _appeal_actions[pid] = {}
    _appeal_actions[pid]["status"] = "appeal_resubmitted"
    _appeal_actions[pid]["last_updated"] = now

    _save_statuses(statuses)

    return {
        "success": True,
        "message": "Appeal resubmitted successfully",
        "submission_id": f"SUB-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "estimated_response_time": "3-5 business days",
    }
