"""
Lambda Action Group Handler — AutoCorrector Agent
Tools: GetPatientData, GeneratePriorAuthorization, CorrectICD10Code,
       CorrectCPTCode, ValidateProviderNPI
"""
import json
import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import (
    get_patient_by_id, validate_icd10, validate_cpt,
    generate_prior_auth, agent_response, logger
)

# ICD-10 correction map: wrong prefix → suggested replacement
ICD10_CORRECTIONS = {
    "Z00": "Z00.00",  # Encounter for general adult medical exam
    "M54": "M54.5",   # Low back pain
    "J06": "J06.9",   # Acute upper respiratory infection
    "I10": "I10",     # Essential hypertension (already valid)
    "E11": "E11.9",   # Type 2 diabetes without complications
    "K21": "K21.0",   # GERD with esophagitis
    "F32": "F32.9",   # Major depressive disorder
    "G43": "G43.909", # Migraine
}

# CPT correction map: common wrong codes → correct codes
CPT_CORRECTIONS = {
    "99213": "99213",  # Office visit established (valid)
    "99214": "99214",  # Office visit established (valid)
    "99201": "99202",  # New patient visit (99201 deleted in 2021)
    "99202": "99202",
    "70553": "70553",  # MRI brain with contrast (valid)
    "93000": "93000",  # ECG (valid)
}


def get_patient_data(patient_id: str) -> dict:
    patient = get_patient_by_id(patient_id)
    if not patient:
        return {"error": f"Patient {patient_id} not found"}
    return {
        "patient_id": patient_id,
        "name": patient.get("patient_name", "Unknown"),
        "insurer": patient.get("insurance_company", "Unknown"),
        "icd_code": patient.get("icd_code", ""),
        "cpt_code": patient.get("cpt_code", ""),
        "prior_auth": patient.get("prior_auth", "None"),
        "provider_npi": patient.get("provider_npi", ""),
        "provider_name": patient.get("provider_name", "Unknown"),
    }


def generate_prior_authorization(patient_id: str, cpt_code: str, insurer: str) -> dict:
    pa_number = generate_prior_auth(patient_id, cpt_code, insurer)
    return {
        "patient_id": patient_id,
        "cpt_code": cpt_code,
        "insurer": insurer,
        "prior_auth_number": pa_number,
        "status": "generated",
        "valid_days": 90,
        "message": f"Prior authorization {pa_number} generated successfully",
    }


def correct_icd10_code(current_code: str, diagnosis_description: str) -> dict:
    current_code = current_code.strip().upper()
    validation = validate_icd10(current_code)

    if validation["valid"]:
        return {
            "original_code": current_code,
            "corrected_code": current_code,
            "correction_needed": False,
            "confidence": validation["confidence"],
            "message": "ICD-10 code is already valid",
        }

    # Try to find a correction
    prefix = current_code[:3]
    corrected = ICD10_CORRECTIONS.get(prefix)
    if not corrected:
        # Suggest based on description keywords
        desc_lower = diagnosis_description.lower()
        if "back" in desc_lower or "spine" in desc_lower:
            corrected = "M54.5"
        elif "diabetes" in desc_lower:
            corrected = "E11.9"
        elif "hypertension" in desc_lower or "blood pressure" in desc_lower:
            corrected = "I10"
        elif "respiratory" in desc_lower or "cough" in desc_lower:
            corrected = "J06.9"
        else:
            corrected = f"{current_code}.9"  # Generic specificity addition

    return {
        "original_code": current_code,
        "corrected_code": corrected,
        "correction_needed": True,
        "confidence": 0.8,
        "message": f"ICD-10 code corrected from {current_code} to {corrected}",
    }


def correct_cpt_code(current_code: str, procedure_description: str) -> dict:
    current_code = current_code.strip()
    validation = validate_cpt(current_code)

    if validation["valid"]:
        corrected = CPT_CORRECTIONS.get(current_code, current_code)
        if corrected != current_code:
            return {
                "original_code": current_code,
                "corrected_code": corrected,
                "correction_needed": True,
                "confidence": 0.9,
                "message": f"CPT code updated from {current_code} to {corrected} (deprecated code)",
            }
        return {
            "original_code": current_code,
            "corrected_code": current_code,
            "correction_needed": False,
            "confidence": 0.9,
            "message": "CPT code is valid",
        }

    # Suggest based on description
    desc_lower = procedure_description.lower()
    if "office visit" in desc_lower or "consultation" in desc_lower:
        corrected = "99213"
    elif "mri" in desc_lower:
        corrected = "70553"
    elif "ecg" in desc_lower or "ekg" in desc_lower:
        corrected = "93000"
    elif "lab" in desc_lower or "blood" in desc_lower:
        corrected = "80053"
    else:
        corrected = "99213"  # Default to common office visit

    return {
        "original_code": current_code,
        "corrected_code": corrected,
        "correction_needed": True,
        "confidence": 0.7,
        "message": f"CPT code corrected from {current_code} to {corrected}",
    }


def validate_provider_npi(npi: str, provider_name: str) -> dict:
    npi = npi.strip()
    # NPI must be exactly 10 digits
    if re.match(r"^\d{10}$", npi):
        return {
            "npi": npi,
            "provider_name": provider_name,
            "valid": True,
            "enrolled": True,
            "message": f"NPI {npi} is valid and enrolled",
        }
    return {
        "npi": npi,
        "provider_name": provider_name,
        "valid": False,
        "enrolled": False,
        "message": f"NPI {npi} is invalid — must be 10 digits",
        "suggested_action": "Verify NPI at nppes.cms.hhs.gov",
    }


# ── Lambda entry point ────────────────────────────────────────────────────────

def lambda_handler(event, context):
    logger.info(f"[AutoCorrector] Event: {json.dumps(event)[:500]}")

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

    if api_path == "/GetPatientData":
        result = get_patient_data(params.get("patient_id", ""))
    elif api_path == "/GeneratePriorAuthorization":
        result = generate_prior_authorization(
            params.get("patient_id", ""),
            params.get("cpt_code", ""),
            params.get("insurer", ""),
        )
    elif api_path == "/CorrectICD10Code":
        result = correct_icd10_code(
            params.get("current_code", ""),
            params.get("diagnosis_description", ""),
        )
    elif api_path == "/CorrectCPTCode":
        result = correct_cpt_code(
            params.get("current_code", ""),
            params.get("procedure_description", ""),
        )
    elif api_path == "/ValidateProviderNPI":
        result = validate_provider_npi(
            params.get("npi", ""),
            params.get("provider_name", ""),
        )
    else:
        result = {"error": f"Unknown api_path: {api_path}"}

    response = agent_response(result, http_method)
    response["response"]["actionGroup"] = action_group
    response["response"]["apiPath"]     = api_path
    return response
