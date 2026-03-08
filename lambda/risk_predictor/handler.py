"""
Lambda Action Group Handler — RiskPredictor Agent
Tools: GetPatientData, ValidateICD10Code, ValidateCPTCode,
       CheckPriorAuthorization, AnalyzeDenialPatterns
"""
import json
import sys
import os

# Allow importing shared utils when packaged with Lambda
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import (
    get_patient_by_id, validate_icd10, validate_cpt,
    generate_prior_auth, load_denial_patterns,
    agent_response, logger
)


# ── Tool implementations ──────────────────────────────────────────────────────

def get_patient_data(patient_id: str) -> dict:
    patient = get_patient_by_id(patient_id)
    if not patient:
        return {"error": f"Patient {patient_id} not found", "patient_id": patient_id}
    return {
        "patient_id": patient_id,
        "name": patient.get("patient_name", "Unknown"),
        "age": patient.get("age", "Unknown"),
        "gender": patient.get("gender", "Unknown"),
        "insurer": patient.get("insurance_company", "Unknown"),
        "diagnosis": patient.get("diagnosis", "Unknown"),
        "icd_code": patient.get("icd_code", ""),
        "cpt_code": patient.get("cpt_code", ""),
        "claim_amount": patient.get("claim_amount", 0),
        "medical_history": patient.get("medical_history", "None"),
        "prior_auth": patient.get("prior_auth", "None"),
        "provider": patient.get("provider_name", "Unknown"),
        "provider_npi": patient.get("provider_npi", ""),
    }


def validate_icd10_code(icd10_code: str) -> dict:
    return validate_icd10(icd10_code)


def validate_cpt_code(cpt_code: str) -> dict:
    return validate_cpt(cpt_code)


def check_prior_authorization(patient_id: str, cpt_code: str, insurer: str) -> dict:
    patient = get_patient_by_id(patient_id)
    existing_pa = (patient or {}).get("prior_auth", "")
    if existing_pa and existing_pa not in ("None", "N/A", ""):
        return {
            "has_prior_auth": True,
            "prior_auth_number": existing_pa,
            "status": "valid",
            "message": f"Prior authorization {existing_pa} found",
        }
    # Determine if PA is required based on CPT range
    try:
        cpt_num = int(cpt_code)
        pa_required = cpt_num in range(70000, 79999) or cpt_num in range(90000, 99199)
    except ValueError:
        pa_required = False
    return {
        "has_prior_auth": False,
        "prior_auth_required": pa_required,
        "status": "missing" if pa_required else "not_required",
        "message": "Prior authorization required but not found" if pa_required else "Prior authorization not required for this procedure",
        "generated_pa": generate_prior_auth(patient_id, cpt_code, insurer) if pa_required else None,
    }


def analyze_denial_patterns(insurer: str, cpt_code: str, icd_code: str) -> dict:
    patterns = load_denial_patterns()
    insurer_key = next((k for k in patterns if k.lower() in insurer.lower()), None)
    insurer_patterns = patterns.get(insurer_key, patterns.get("DEFAULT", []))

    # Find patterns matching this CPT/ICD combo
    matching = [p for p in insurer_patterns if
                p.get("cpt_code") == cpt_code or p.get("icd_code") == icd_code]

    denial_rate = len(matching) / max(len(insurer_patterns), 1)
    risk_score = min(0.9, denial_rate + 0.2) if matching else 0.3

    return {
        "insurer": insurer,
        "cpt_code": cpt_code,
        "icd_code": icd_code,
        "historical_denial_rate": round(denial_rate, 2),
        "risk_score": round(risk_score, 2),
        "matching_patterns": len(matching),
        "top_denial_reasons": [p.get("reason", "") for p in matching[:3]],
        "recommendation": "High denial risk — review documentation" if risk_score > 0.6 else "Moderate risk — proceed with standard documentation",
    }


# ── Lambda entry point ────────────────────────────────────────────────────────

def lambda_handler(event, context):
    logger.info(f"[RiskPredictor] Event: {json.dumps(event)[:500]}")

    action_group = event.get("actionGroup", "")
    api_path     = event.get("apiPath", "")
    http_method  = event.get("httpMethod", "GET")
    parameters   = event.get("parameters", [])
    request_body = event.get("requestBody", {})

    # Parse parameters into a flat dict
    params = {p["name"]: p["value"] for p in parameters}

    # Also check requestBody for POST payloads
    if request_body:
        try:
            body_content = request_body.get("content", {})
            json_body = body_content.get("application/json", {}).get("properties", [])
            for prop in json_body:
                params[prop["name"]] = prop["value"]
        except Exception:
            pass

    logger.info(f"[RiskPredictor] api_path={api_path} params={params}")

    # Route to correct tool
    if api_path == "/GetPatientData":
        result = get_patient_data(params.get("patient_id", ""))
    elif api_path == "/ValidateICD10Code":
        result = validate_icd10_code(params.get("icd10_code", ""))
    elif api_path == "/ValidateCPTCode":
        result = validate_cpt_code(params.get("cpt_code", ""))
    elif api_path == "/CheckPriorAuthorization":
        result = check_prior_authorization(
            params.get("patient_id", ""),
            params.get("cpt_code", ""),
            params.get("insurer", ""),
        )
    elif api_path == "/AnalyzeDenialPatterns":
        result = analyze_denial_patterns(
            params.get("insurer", ""),
            params.get("cpt_code", ""),
            params.get("icd_code", ""),
        )
    else:
        result = {"error": f"Unknown api_path: {api_path}"}

    response = agent_response(result, http_method)
    response["response"]["actionGroup"] = action_group
    response["response"]["apiPath"]     = api_path
    return response
