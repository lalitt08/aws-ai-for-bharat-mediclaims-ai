"""
Lambda Action Group Handler — AppealGenerator Agent
Tools: GetDenialDetails, GenerateAppealLetter, SaveAppealToS3, CheckAppealRequirements
"""
import json
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import (
    get_patient_by_id, load_claim_status,
    s3_read_json, s3_client, agent_response, logger,
    BUCKET
)

# Denial code → required documents mapping
DENIAL_REQUIREMENTS = {
    "CO-16": {
        "required_docs": ["Clinical notes", "Lab results", "Physician statement", "Medical necessity letter"],
        "appeal_type": "clinical_documentation",
        "success_rate": 0.82,
    },
    "CO-197": {
        "required_docs": ["New prior authorization", "Updated treatment plan", "Physician order"],
        "appeal_type": "prior_auth",
        "success_rate": 0.75,
    },
    "CO-4": {
        "required_docs": ["Corrected ICD-10 code", "Medical necessity letter", "Diagnosis documentation"],
        "appeal_type": "coding_correction",
        "success_rate": 0.78,
    },
    "CO-50": {
        "required_docs": ["Medical necessity documentation", "Clinical guidelines reference", "Peer-reviewed literature"],
        "appeal_type": "medical_necessity",
        "success_rate": 0.65,
    },
    "CO-11": {
        "required_docs": ["Corrected diagnosis code", "Clinical documentation"],
        "appeal_type": "coding_correction",
        "success_rate": 0.80,
    },
    "DEFAULT": {
        "required_docs": ["Clinical documentation", "Medical records", "Physician statement"],
        "appeal_type": "general",
        "success_rate": 0.70,
    },
}


def get_denial_details(patient_id: str, claim_id: str) -> dict:
    all_claims = load_claim_status()
    entry = all_claims.get(patient_id, {})

    if not entry:
        # Try searching by claim_id
        for pid, e in all_claims.items():
            if e.get("claim_id") == claim_id:
                entry = e
                patient_id = pid
                break

    denial_info = entry.get("denial_info", {})
    return {
        "patient_id": patient_id,
        "claim_id": claim_id,
        "status": entry.get("status", "unknown"),
        "denial_reason": denial_info.get("reason", entry.get("denial_reason", "Not specified")),
        "denial_code": denial_info.get("code", "CO-16"),
        "denial_details": denial_info.get("details", ""),
        "required_items": denial_info.get("required_items", []),
        "insurer": entry.get("insurer", ""),
        "claim_amount": entry.get("claim_amount", 0),
    }


def check_appeal_requirements(denial_code: str, denial_reason: str) -> dict:
    req = DENIAL_REQUIREMENTS.get(denial_code, DENIAL_REQUIREMENTS["DEFAULT"])
    return {
        "denial_code": denial_code,
        "denial_reason": denial_reason,
        "required_documents": req["required_docs"],
        "appeal_type": req["appeal_type"],
        "estimated_success_rate": req["success_rate"],
        "deadline_days": 30,
        "submission_method": "electronic",
    }


def generate_appeal_letter(claim_data_json: str, denial_reason: str, denial_code: str) -> dict:
    """Generate appeal letter using Bedrock LLM (bearer token)."""
    try:
        claim = json.loads(claim_data_json) if isinstance(claim_data_json, str) else claim_data_json
    except Exception:
        claim = {}

    patient_name  = claim.get("patient_name", "Patient")
    insurer       = claim.get("insurer", claim.get("insurance_company", "Insurance Company"))
    claim_id      = claim.get("claim_id", "N/A")
    cpt_code      = claim.get("cpt_code", claim.get("procedure_code", "N/A"))
    icd_code      = claim.get("icd_code", claim.get("diagnosis_code", "N/A"))
    provider      = claim.get("provider", claim.get("provider_name", "Provider"))
    service_date  = claim.get("service_date", datetime.utcnow().strftime("%Y-%m-%d"))
    amount        = claim.get("claim_amount", 0)

    # Try Bedrock LLM via boto3 (IAM key auth)
    appeal_text = None
    try:
        import boto3
        bedrock = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
        model_id = os.environ.get("AWS_BEDROCK_MODEL_ID", "us.amazon.nova-micro-v1:0")

        prompt_text = (
            f"Write a formal medical insurance appeal letter.\n\n"
            f"Patient: {patient_name}\nInsurer: {insurer}\nClaim ID: {claim_id}\n"
            f"Service Date: {service_date}\nAmount: ${amount}\n"
            f"CPT Code: {cpt_code}\nICD-10: {icd_code}\nProvider: {provider}\n"
            f"Denial Reason: {denial_reason} (Code: {denial_code})\n\n"
            f"Write a complete, professional appeal letter that:\n"
            f"1. States the reason for appeal\n"
            f"2. Provides medical justification\n"
            f"3. References relevant clinical guidelines\n"
            f"4. Requests reconsideration with supporting documentation\n"
            f"Keep it under 400 words."
        )

        payload = json.dumps({
            "messages": [{"role": "user", "content": [{"text": prompt_text}]}],
            "inferenceConfig": {"maxTokens": 600, "temperature": 0.3},
        })

        resp = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=payload,
        )
        body = json.loads(resp["body"].read())
        appeal_text = body["output"]["message"]["content"][0]["text"].strip()
    except Exception as e:
        logger.warning(f"[AppealGenerator] Bedrock LLM call failed: {e}")

    # Fallback: structured template
    if not appeal_text:
        req_info = DENIAL_REQUIREMENTS.get(denial_code, DENIAL_REQUIREMENTS["DEFAULT"])
        appeal_text = (
            f"RE: Appeal for Claim {claim_id} — {patient_name}\n\n"
            f"Dear {insurer} Appeals Department,\n\n"
            f"We are writing to formally appeal the denial of claim {claim_id} for patient {patient_name}, "
            f"dated {service_date}, for services rendered by {provider}.\n\n"
            f"Denial Reason: {denial_reason} (Code: {denial_code})\n\n"
            f"The procedure (CPT {cpt_code}) was medically necessary for the treatment of the patient's "
            f"condition (ICD-10: {icd_code}). We respectfully request reconsideration based on the "
            f"attached clinical documentation.\n\n"
            f"Supporting documents enclosed: {', '.join(req_info['required_docs'])}.\n\n"
            f"We request a response within 30 days. Total claim amount: ${amount}.\n\n"
            f"Sincerely,\n{provider}\n"
        )

    return {
        "appeal_text": appeal_text,
        "claim_id": claim_id,
        "patient_name": patient_name,
        "insurer": insurer,
        "denial_code": denial_code,
        "generated_at": datetime.utcnow().isoformat(),
        "word_count": len(appeal_text.split()),
    }


def save_appeal_to_s3(patient_id: str, claim_id: str, appeal_text: str) -> dict:
    key = f"appeals/appeal_{claim_id}_{patient_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.txt"
    try:
        s3_client().put_object(
            Bucket=BUCKET,
            Key=key,
            Body=appeal_text.encode("utf-8"),
            ContentType="text/plain",
        )
        return {
            "saved": True,
            "s3_key": key,
            "s3_bucket": BUCKET,
            "patient_id": patient_id,
            "claim_id": claim_id,
            "message": f"Appeal saved to S3: {key}",
        }
    except Exception as e:
        logger.error(f"[AppealGenerator] S3 save failed: {e}")
        return {"saved": False, "error": str(e)}


# ── Lambda entry point ────────────────────────────────────────────────────────

def lambda_handler(event, context):
    logger.info(f"[AppealGenerator] Event: {json.dumps(event)[:500]}")

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

    if api_path == "/GetDenialDetails":
        result = get_denial_details(
            params.get("patient_id", ""),
            params.get("claim_id", ""),
        )
    elif api_path == "/CheckAppealRequirements":
        result = check_appeal_requirements(
            params.get("denial_code", "CO-16"),
            params.get("denial_reason", ""),
        )
    elif api_path == "/GenerateAppealLetter":
        result = generate_appeal_letter(
            params.get("claim_data_json", "{}"),
            params.get("denial_reason", ""),
            params.get("denial_code", "CO-16"),
        )
    elif api_path == "/SaveAppealToS3":
        result = save_appeal_to_s3(
            params.get("patient_id", ""),
            params.get("claim_id", ""),
            params.get("appeal_text", ""),
        )
    else:
        result = {"error": f"Unknown api_path: {api_path}"}

    response = agent_response(result, http_method)
    response["response"]["actionGroup"] = action_group
    response["response"]["apiPath"]     = api_path
    return response
