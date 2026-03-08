"""
Lambda Action Group Handler — FeedbackLearner Agent
Tools: RecordClaimOutcome, UpdateDenialPatterns, GetLearningInsights
"""
import json
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import (
    s3_append_jsonl, s3_read_json, s3_write_json,
    load_denial_patterns, agent_response, logger
)


def record_claim_outcome(patient_id: str, claim_id: str, outcome: str,
                          insurer: str, cpt_code: str, icd_code: str) -> dict:
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "patient_id": patient_id,
        "claim_id": claim_id,
        "outcome": outcome,
        "insurer": insurer,
        "cpt_code": cpt_code,
        "icd_code": icd_code,
    }
    s3_append_jsonl("logs/claim_outcomes.jsonl", record)
    return {
        "recorded": True,
        "claim_id": claim_id,
        "outcome": outcome,
        "message": f"Outcome '{outcome}' recorded for claim {claim_id}",
    }


def update_denial_patterns(insurer: str, cpt_code: str,
                            denial_reason: str, was_appealed_successfully: str) -> dict:
    """Update denial_patterns.json in S3 with new outcome data."""
    patterns = load_denial_patterns()
    appealed_ok = str(was_appealed_successfully).lower() in ("true", "1", "yes")

    # Find or create insurer entry
    insurer_key = next((k for k in patterns if k.lower() in insurer.lower()), insurer)
    if insurer_key not in patterns:
        patterns[insurer_key] = []

    # Find existing pattern for this CPT
    existing = next((p for p in patterns[insurer_key] if p.get("cpt_code") == cpt_code), None)

    if existing:
        existing["total_claims"] = existing.get("total_claims", 1) + 1
        if appealed_ok:
            existing["successful_appeals"] = existing.get("successful_appeals", 0) + 1
        existing["success_rate"] = round(
            existing.get("successful_appeals", 0) / existing["total_claims"], 2
        )
        existing["last_updated"] = datetime.utcnow().isoformat()
    else:
        patterns[insurer_key].append({
            "cpt_code": cpt_code,
            "reason": denial_reason,
            "total_claims": 1,
            "successful_appeals": 1 if appealed_ok else 0,
            "success_rate": 1.0 if appealed_ok else 0.0,
            "last_updated": datetime.utcnow().isoformat(),
        })

    s3_write_json("claims/denial_patterns.json", patterns)
    return {
        "updated": True,
        "insurer": insurer,
        "cpt_code": cpt_code,
        "denial_reason": denial_reason,
        "appeal_successful": appealed_ok,
        "message": "Denial patterns updated in S3",
    }


def get_learning_insights(insurer: str, cpt_code: str) -> dict:
    patterns = load_denial_patterns()
    insurer_key = next((k for k in patterns if k.lower() in insurer.lower()), None)

    if not insurer_key:
        return {
            "insurer": insurer,
            "cpt_code": cpt_code,
            "insights": "No historical data available for this insurer",
            "recommendation": "Proceed with standard documentation",
            "historical_success_rate": None,
        }

    insurer_patterns = patterns[insurer_key]
    cpt_pattern = next((p for p in insurer_patterns if p.get("cpt_code") == cpt_code), None)

    if not cpt_pattern:
        # Return aggregate insurer stats
        total = sum(p.get("total_claims", 0) for p in insurer_patterns)
        appeals = sum(p.get("successful_appeals", 0) for p in insurer_patterns)
        agg_rate = round(appeals / max(total, 1), 2)
        return {
            "insurer": insurer,
            "cpt_code": cpt_code,
            "insights": f"No specific data for CPT {cpt_code}. Insurer aggregate success rate: {agg_rate:.0%}",
            "recommendation": "Use standard appeal process",
            "historical_success_rate": agg_rate,
            "data_points": total,
        }

    success_rate = cpt_pattern.get("success_rate", 0)
    recommendation = (
        "High success rate — proceed with appeal" if success_rate > 0.75
        else "Moderate success — ensure complete documentation"
        if success_rate > 0.5
        else "Low success rate — consider peer-to-peer review"
    )

    return {
        "insurer": insurer,
        "cpt_code": cpt_code,
        "historical_success_rate": success_rate,
        "total_claims": cpt_pattern.get("total_claims", 0),
        "successful_appeals": cpt_pattern.get("successful_appeals", 0),
        "top_denial_reason": cpt_pattern.get("reason", "Unknown"),
        "recommendation": recommendation,
        "last_updated": cpt_pattern.get("last_updated", ""),
    }


# ── Lambda entry point ────────────────────────────────────────────────────────

def lambda_handler(event, context):
    logger.info(f"[FeedbackLearner] Event: {json.dumps(event)[:500]}")

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

    if api_path == "/RecordClaimOutcome":
        result = record_claim_outcome(
            params.get("patient_id", ""),
            params.get("claim_id", ""),
            params.get("outcome", ""),
            params.get("insurer", ""),
            params.get("cpt_code", ""),
            params.get("icd_code", ""),
        )
    elif api_path == "/UpdateDenialPatterns":
        result = update_denial_patterns(
            params.get("insurer", ""),
            params.get("cpt_code", ""),
            params.get("denial_reason", ""),
            params.get("was_appealed_successfully", "false"),
        )
    elif api_path == "/GetLearningInsights":
        result = get_learning_insights(
            params.get("insurer", ""),
            params.get("cpt_code", ""),
        )
    else:
        result = {"error": f"Unknown api_path: {api_path}"}

    response = agent_response(result, http_method)
    response["response"]["actionGroup"] = action_group
    response["response"]["apiPath"]     = api_path
    return response
