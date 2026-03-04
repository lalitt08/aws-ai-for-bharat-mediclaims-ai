"""
Metrics API endpoints - Uses REAL claim data from claim_status.json
"""

from fastapi import APIRouter, Query
from typing import Dict, Any
from datetime import datetime
import os, json

metrics_router = APIRouter()

_PRE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLAIM_FILE = os.path.join(_PRE_DIR, "data", "claim_status.json")

def _load():
    try:
        if os.path.exists(_CLAIM_FILE):
            with open(_CLAIM_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


@metrics_router.get("/dashboard-summary")
async def get_dashboard_summary():
    """Dashboard metrics computed from real claim data."""
    statuses = _load()
    total = len(statuses)
    approved = 0
    denied = 0
    appealed = 0
    resubmitted = 0
    total_recovered = 0.0
    total_at_risk = 0.0
    processing_times = []

    for entry in statuses.values():
        s = (entry.get("status") or "").lower()
        sub = entry.get("submission_result") or {}

        if s == "approved":
            approved += 1
            total_recovered += float(sub.get("approved_amount", 0) or 0)
        else:
            denied += 1
            # Estimate amount at risk from denial info
            denial_info = sub.get("denial_info") or {}

        if "appeal" in s:
            appealed += 1
        if "resubmit" in s:
            resubmitted += 1

        pt = entry.get("processing_time")
        if pt:
            processing_times.append(float(pt))

    avg_time = round(sum(processing_times) / len(processing_times), 1) if processing_times else 0
    success_rate = round(approved / total * 100, 1) if total else 0

    return {
        "total_claims": total,
        "approved_claims": approved,
        "denied_claims": denied,
        "appealed_claims": appealed,
        "resubmitted_claims": resubmitted,
        "success_rate": success_rate,
        "average_processing_time": avg_time,
        "total_recovered": round(total_recovered, 2),
        "last_updated": datetime.now().isoformat(),
    }


@metrics_router.get("/denial-reasons-analysis")
async def get_denial_reasons_analysis():
    """Denial reason analysis from real data."""
    statuses = _load()
    reason_data = {}

    for entry in statuses.values():
        sub = entry.get("submission_result") or {}
        denial_info = sub.get("denial_info") or {}
        reason = denial_info.get("reason", "")
        if not reason:
            continue
        if reason not in reason_data:
            reason_data[reason] = {"count": 0, "success_rates": []}
        reason_data[reason]["count"] += 1
        sr = denial_info.get("success_rate")
        if sr is not None:
            reason_data[reason]["success_rates"].append(float(sr))

    total = sum(d["count"] for d in reason_data.values()) or 1
    denial_reasons = []
    for reason, data in sorted(reason_data.items(), key=lambda x: x[1]["count"], reverse=True):
        avg_sr = sum(data["success_rates"]) / len(data["success_rates"]) if data["success_rates"] else 0.5
        denial_reasons.append({
            "reason": reason,
            "count": data["count"],
            "percentage": round(data["count"] / total * 100, 1),
            "avg_appeal_success": round(avg_sr, 2),
        })

    return {
        "analysis_date": datetime.now().isoformat(),
        "total_denials_analyzed": total,
        "denial_reasons": denial_reasons,
    }


@metrics_router.get("/payer-performance")
async def get_payer_performance():
    """Payer performance from real data."""
    statuses = _load()
    # We need patient CSV for insurer info
    import csv
    csv_file = os.path.join(_PRE_DIR, "data", "patients1.csv")
    patients = {}
    try:
        if os.path.exists(csv_file):
            with open(csv_file, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    pid = row.get("patient_id", "").strip()
                    if pid:
                        patients[pid] = row
    except Exception:
        pass

    payer_stats: Dict[str, Dict[str, Any]] = {}
    for pid, entry in statuses.items():
        csv_row = patients.get(pid, {})
        payer = csv_row.get("insurer", "Unknown")
        s = (entry.get("status") or "").lower()
        sub = entry.get("submission_result") or {}

        if payer not in payer_stats:
            payer_stats[payer] = {"total": 0, "denied": 0, "approved": 0, "appealed": 0, "recovered": 0.0, "times": []}

        payer_stats[payer]["total"] += 1
        if s == "approved":
            payer_stats[payer]["approved"] += 1
            payer_stats[payer]["recovered"] += float(sub.get("approved_amount", 0) or 0)
        else:
            payer_stats[payer]["denied"] += 1
        if "appeal" in s or "resubmit" in s:
            payer_stats[payer]["appealed"] += 1
        pt = entry.get("processing_time")
        if pt:
            payer_stats[payer]["times"].append(float(pt))

    payers = []
    for name, stats in payer_stats.items():
        denial_rate = stats["denied"] / stats["total"] if stats["total"] else 0
        appeal_success = stats["appealed"] / stats["denied"] if stats["denied"] else 0
        avg_time = sum(stats["times"]) / len(stats["times"]) if stats["times"] else 0
        payers.append({
            "payer_name": name,
            "total_claims": stats["total"],
            "denied_claims": stats["denied"],
            "denial_rate": round(denial_rate, 2),
            "appeals_submitted": stats["appealed"],
            "appeal_success_rate": round(min(appeal_success, 1.0), 2),
            "avg_processing_time": round(avg_time, 1),
            "revenue_recovered": round(stats["recovered"], 2),
        })

    return {"analysis_period": "Current data", "payer_performance": payers}


@metrics_router.get("/performance-kpis")
async def get_performance_kpis():
    """KPIs computed from real data."""
    statuses = _load()
    total = len(statuses)
    approved = sum(1 for e in statuses.values() if (e.get("status") or "").lower() == "approved")
    denied = total - approved
    appealed = sum(1 for e in statuses.values() if "appeal" in (e.get("status") or "").lower() or "resubmit" in (e.get("status") or "").lower())
    times = [float(e["processing_time"]) for e in statuses.values() if e.get("processing_time")]
    avg_time = round(sum(times) / len(times), 1) if times else 0
    success_rate = round(approved / total, 3) if total else 0
    appeal_rate = round(appealed / denied, 3) if denied else 0

    # Quality scores from data
    quality_scores = [float((e.get("submission_result") or {}).get("data_quality_score", 50)) for e in statuses.values() if (e.get("submission_result") or {}).get("data_quality_score")]
    avg_quality = round(sum(quality_scores) / len(quality_scores) / 100, 3) if quality_scores else 0.5

    return {
        "operational_kpis": {
            "claim_success_rate": {"current": success_rate, "target": 0.800, "trend": "tracking"},
            "average_processing_time": {"current": avg_time, "target": 15.0, "unit": "seconds", "trend": "tracking"},
            "appeal_action_rate": {"current": appeal_rate, "target": 0.900, "trend": "tracking"},
        },
        "quality_kpis": {
            "avg_data_quality_score": {"current": avg_quality, "target": 0.900, "trend": "tracking"},
            "total_claims_processed": {"current": total, "trend": "tracking"},
            "denied_claims": {"current": denied, "trend": "tracking"},
        },
    }


@metrics_router.get("/real-time-stats")
async def get_real_time_stats():
    """Real-time stats from actual data."""
    statuses = _load()
    total = len(statuses)
    approved = sum(1 for e in statuses.values() if (e.get("status") or "").lower() == "approved")
    denied = total - approved
    appealed = sum(1 for e in statuses.values() if "appeal" in (e.get("status") or "").lower())
    awaiting = sum(1 for e in statuses.values() if "awaiting" in (e.get("status") or "").lower())

    recovered = sum(float((e.get("submission_result") or {}).get("approved_amount", 0) or 0) for e in statuses.values())

    return {
        "timestamp": datetime.now().isoformat(),
        "live_stats": {
            "total_claims": total,
            "approved_claims": approved,
            "denied_claims": denied,
            "appeals_in_progress": appealed,
            "awaiting_action": awaiting,
            "revenue_recovered": round(recovered, 2),
        },
    }
