"""
Lambda Action Group Handler — ClaimSubmitter Agent
Tools: CheckEligibility, SubmitClaimToInsurer, GetClaimStatus, SaveClaimResult
"""
import json
import sys
import os
import uuid
import random
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import (
    get_patient_by_id, get_denial_for_insurer,
    load_claim_status, update_claim_entry,
    s3_write_json, agent_response, logger,
    PRIMARY_API, SECONDARY_API
)

# Insurers routed to secondary API
SECONDARY_INSURERS = {"cigna", "united", "unitedhealth", "humana"}


def check_eligibility(patient_id: str, insurer: str, service_date: str) -> dict:
    patient = get_patient_by_id(patient_id)
    if not patient:
        return {
            "eligible": False,
            "patient_id": patient_id,
            "reason": "Patient not found in system",
        }

    # Simulate eligibility check — in production this calls a real payer API
    patient_insurer = patient.get("insurance_company", "").lower()
    insurer_lower   = insurer.lower()

    # Check if insurer matches patient's plan
    insurer_match = any(word in patient_insurer for word in insurer_lower.split())

    return {
        "eligible": True,
        "patient_id": patient_id,
        "insurer": insurer,
        "insurer_match": insurer_match,
        "coverage_active": True,
        "deductible_met": random.choice([True, False]),
        "copay": random.choice([20, 30, 40, 50]),
        "service_date": service_date,
        "message": "Patient is eligible for coverage",
    }


def submit_claim_to_insurer(claim_data_json: str, insurer: str) -> dict:
    """Route claim to primary or secondary insurer API and return result."""
    try:
        claim = json.loads(claim_data_json) if isinstance(claim_data_json, str) else claim_data_json
    except Exception:
        claim = {}

    insurer_lower = insurer.lower()
    api_url = SECONDARY_API if any(k in insurer_lower for k in SECONDARY_INSURERS) else PRIMARY_API

    # Try real HTTP call to insurer API
    try:
        import urllib.request
        payload = json.dumps(claim).encode("utf-8")
        req = urllib.request.Request(
            f"{api_url}/submit",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return {
                "claim_id": result.get("claim_id", claim.get("claim_id", f"CLM-{uuid.uuid4().hex[:8].upper()}")),
                "status": result.get("status", "submitted"),
                "insurer": insurer,
                "api_used": api_url,
                "denial_info": result.get("denial_info"),
                "message": result.get("message", "Claim submitted successfully"),
            }
    except Exception as e:
        logger.warning(f"[ClaimSubmitter] Insurer API unreachable ({api_url}): {e} — using simulation")

    # Simulation fallback (for demo / when insurer API is down)
    claim_id = claim.get("claim_id") or f"CLM-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{claim.get('patient_id', 'UNK')}"
    amount = float(claim.get("claim_amount", 0))

    # Simulate approval/denial based on amount and insurer
    denial_threshold = 800 if "cigna" in insurer_lower else 1000
    if amount > denial_threshold:
        denial = get_denial_for_insurer(insurer)
        return {
            "claim_id": claim_id,
            "status": "denied",
            "insurer": insurer,
            "api_used": api_url,
            "denial_info": denial,
            "message": f"Claim denied: {denial['reason']}",
        }

    return {
        "claim_id": claim_id,
        "status": "approved",
        "insurer": insurer,
        "api_used": api_url,
        "approved_amount": amount,
        "message": "Claim approved",
    }


def get_claim_status(claim_id: str) -> dict:
    all_claims = load_claim_status()
    # Search by claim_id across all patients
    for patient_id, entry in all_claims.items():
        if entry.get("claim_id") == claim_id or patient_id == claim_id:
            return {"found": True, "claim_id": claim_id, "patient_id": patient_id, **entry}
    return {"found": False, "claim_id": claim_id, "message": "Claim not found"}


def save_claim_result(patient_id: str, claim_id: str, result_json: str) -> dict:
    try:
        result = json.loads(result_json) if isinstance(result_json, str) else result_json
    except Exception:
        result = {"raw": str(result_json)}

    entry = {
        "claim_id": claim_id,
        "status": result.get("status", "unknown"),
        "insurer": result.get("insurer", ""),
        "denial_info": result.get("denial_info"),
        "approved_amount": result.get("approved_amount"),
        "submitted_at": datetime.utcnow().isoformat(),
    }
    update_claim_entry(patient_id, entry)
    return {"saved": True, "patient_id": patient_id, "claim_id": claim_id, "status": entry["status"]}


# ── Lambda entry point ────────────────────────────────────────────────────────

def lambda_handler(event, context):
    logger.info(f"[ClaimSubmitter] Event: {json.dumps(event)[:500]}")

    action_group = event.get("actionGroup", "")
    api_path     = event.get("apiPath", "")
    http_method  = event.get("httpMethod", "GET")
    parameters   = event.get("parameters", [])
    request_body = event.get("requestBody", {})

    params = {p["name"]: p["value"] for p in parameters}
    if request_body:
        try:
            for prop in request_body.get("content", {}).get("application/json", {}).get("properties", []):
                params[prop["name"]] = prop["value"]
        except Exception:
            pass

    if api_path == "/CheckEligibility":
        result = check_eligibility(
            params.get("patient_id", ""),
            params.get("insurer", ""),
            params.get("service_date", ""),
        )
    elif api_path == "/SubmitClaimToInsurer":
        result = submit_claim_to_insurer(
            params.get("claim_data_json", "{}"),
            params.get("insurer", ""),
        )
    elif api_path == "/GetClaimStatus":
        result = get_claim_status(params.get("claim_id", ""))
    elif api_path == "/SaveClaimResult":
        result = save_claim_result(
            params.get("patient_id", ""),
            params.get("claim_id", ""),
            params.get("result_json", "{}"),
        )
    else:
        result = {"error": f"Unknown api_path: {api_path}"}

    response = agent_response(result, http_method)
    response["response"]["actionGroup"] = action_group
    response["response"]["apiPath"]     = api_path
    return response
