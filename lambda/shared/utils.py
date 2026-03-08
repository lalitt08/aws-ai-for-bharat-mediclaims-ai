"""
Shared utilities for all Lambda Action Group handlers.
Provides S3 access, patient data loading, insurer API calls.
"""
import boto3
import json
import os
import csv
import io
import logging
import random
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BUCKET = os.environ.get("S3_BUCKET_NAME", "alpha-claims-demo-390783052961")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
PRIMARY_API   = os.environ.get("PRIMARY_INSURER_API",   "http://localhost:8081")
SECONDARY_API = os.environ.get("SECONDARY_INSURER_API", "http://localhost:8082")


# ── S3 helpers ────────────────────────────────────────────────────────────────

def s3_client():
    return boto3.client("s3", region_name=REGION)


def s3_read_json(key: str) -> dict:
    try:
        obj = s3_client().get_object(Bucket=BUCKET, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception as e:
        logger.warning(f"[S3] read {key} failed: {e}")
        return {}


def s3_write_json(key: str, data: dict):
    try:
        body = json.dumps(data, indent=2, ensure_ascii=True).encode("utf-8")
        s3_client().put_object(Bucket=BUCKET, Key=key, Body=body, ContentType="application/json")
        logger.info(f"[S3] wrote {key}")
    except Exception as e:
        logger.error(f"[S3] write {key} failed: {e}")


def s3_append_jsonl(key: str, record: dict):
    """Append a JSON line to an S3 JSONL file."""
    try:
        existing = ""
        try:
            obj = s3_client().get_object(Bucket=BUCKET, Key=key)
            existing = obj["Body"].read().decode("utf-8")
        except Exception:
            pass
        new_line = json.dumps(record, ensure_ascii=True)
        body = (existing.rstrip("\n") + "\n" + new_line + "\n").encode("utf-8")
        s3_client().put_object(Bucket=BUCKET, Key=key, Body=body, ContentType="application/x-ndjson")
    except Exception as e:
        logger.error(f"[S3] append_jsonl {key} failed: {e}")


def load_claim_status() -> dict:
    return s3_read_json("claims/claim_status.json")


def save_claim_status(data: dict):
    s3_write_json("claims/claim_status.json", data)


def update_claim_entry(patient_id: str, entry: dict):
    all_claims = load_claim_status()
    all_claims[patient_id] = {
        **all_claims.get(patient_id, {}),
        **entry,
        "updated_at": datetime.utcnow().isoformat()
    }
    save_claim_status(all_claims)


# ── Patient data ──────────────────────────────────────────────────────────────

def load_patients_csv() -> List[Dict[str, Any]]:
    """Load patients1.csv from S3."""
    try:
        obj = s3_client().get_object(Bucket=BUCKET, Key="patients/patients1.csv")
        content = obj["Body"].read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(content))
        return [dict(row) for row in reader]
    except Exception as e:
        logger.error(f"[S3] patients CSV load failed: {e}")
        return []


def get_patient_by_id(patient_id: str) -> Optional[Dict[str, Any]]:
    for p in load_patients_csv():
        if p.get("patient_id") == patient_id:
            return p
    return None


def load_denial_patterns() -> dict:
    return s3_read_json("claims/denial_patterns.json")


# ── Denial logic ──────────────────────────────────────────────────────────────

DENIAL_REASONS = {
    "BlueCross": [
        {"reason": "Missing clinical documentation", "code": "CO-16",
         "details": "Clinical notes required for procedure justification",
         "required_items": ["Clinical notes", "Lab results", "Physician statement"],
         "success_rate": 0.82},
        {"reason": "Prior authorization expired", "code": "CO-197",
         "details": "Authorization number not valid for service date",
         "required_items": ["New prior auth number", "Updated treatment plan"],
         "success_rate": 0.75},
    ],
    "Aetna": [
        {"reason": "Diagnosis code mismatch", "code": "CO-4",
         "details": "ICD-10 code does not support medical necessity for CPT",
         "required_items": ["Corrected ICD-10 code", "Medical necessity letter"],
         "success_rate": 0.78},
    ],
    "Cigna": [
        {"reason": "Medical necessity not established", "code": "CO-50",
         "details": "Procedure not medically necessary per clinical guidelines",
         "required_items": ["Medical necessity documentation", "Clinical guidelines reference"],
         "success_rate": 0.65},
    ],
    "United": [
        {"reason": "Incomplete provider credentials", "code": "CO-16",
         "details": "Provider NPI not enrolled with payer",
         "required_items": ["Provider enrollment form", "NPI verification"],
         "success_rate": 0.88},
    ],
    "DEFAULT": [
        {"reason": "Missing clinical documentation", "code": "CO-16",
         "details": "Additional documentation required",
         "required_items": ["Clinical documentation", "Medical records"],
         "success_rate": 0.75},
    ]
}


def get_denial_for_insurer(insurer: str) -> dict:
    insurer_key = next((k for k in DENIAL_REASONS if k.lower() in insurer.lower()), "DEFAULT")
    options = DENIAL_REASONS[insurer_key]
    return random.choice(options)


# ── ICD/CPT validation ────────────────────────────────────────────────────────

VALID_ICD_PREFIXES = ["Z", "M", "J", "I", "E", "K", "F", "G", "N", "R", "S", "T", "L", "H", "C", "D", "B", "A"]
VALID_CPT_RANGES = [(99201, 99499), (70000, 79999), (80000, 89999), (90000, 99199), (10000, 69999)]


def validate_icd10(code: str) -> dict:
    if not code:
        return {"valid": False, "message": "ICD-10 code is empty", "confidence": 0.0}
    code = code.strip().upper()
    if len(code) < 3:
        return {"valid": False, "message": f"ICD-10 code {code} too short", "confidence": 0.1}
    if code[0] in VALID_ICD_PREFIXES:
        return {"valid": True, "message": f"ICD-10 code {code} is valid", "confidence": 0.9, "code": code}
    return {"valid": False, "message": f"ICD-10 code {code} prefix not recognized", "confidence": 0.3}


def validate_cpt(code: str) -> dict:
    if not code:
        return {"valid": False, "message": "CPT code is empty", "confidence": 0.0}
    try:
        num = int(code.strip())
        for lo, hi in VALID_CPT_RANGES:
            if lo <= num <= hi:
                return {"valid": True, "message": f"CPT {code} is valid", "confidence": 0.9, "code": code}
        return {"valid": False, "message": f"CPT {code} out of valid range", "confidence": 0.2}
    except ValueError:
        return {"valid": False, "message": f"CPT {code} is not numeric", "confidence": 0.0}


# ── Prior auth ────────────────────────────────────────────────────────────────

def generate_prior_auth(patient_id: str, cpt_code: str, insurer: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d%H%M")
    return f"PA-{insurer[:3].upper()}-{patient_id[-4:]}-{cpt_code}-{ts}"


# ── Bedrock Agent response builder ───────────────────────────────────────────

def agent_response(body: dict, http_method: str = "GET") -> dict:
    """Format response for Bedrock Agent Action Group."""
    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": "",   # filled by caller
            "apiPath": "",       # filled by caller
            "httpMethod": http_method.upper(),
            "httpStatusCode": 200,
            "responseBody": {
                "application/json": {
                    "body": json.dumps(body, ensure_ascii=True)
                }
            }
        }
    }
