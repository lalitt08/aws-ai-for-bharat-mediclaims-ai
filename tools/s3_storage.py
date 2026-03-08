"""
S3 Storage — centralized read/write for all claim data.
Replaces local file I/O for claim_status.json, patients1.csv, ERA files, appeals.
Falls back to local disk if S3 is unavailable (dev mode).
"""

import boto3
import json
import os
import io
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

BUCKET = os.getenv("S3_BUCKET_NAME", "alpha-claims-demo-390783052961")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Local fallback paths
_LOCAL_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _client():
    return boto3.client("s3", region_name=REGION)


# ── JSON helpers ──────────────────────────────────────────────────────────────

def read_json(s3_key: str, local_fallback: str = None) -> dict:
    """Read a JSON file from S3, fall back to local path."""
    try:
        obj = _client().get_object(Bucket=BUCKET, Key=s3_key)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        logger.info(f"[S3] Read {s3_key}")
        return data
    except Exception as e:
        logger.warning(f"[S3] Read failed for {s3_key}: {e} — using local fallback")
        if local_fallback and os.path.exists(local_fallback):
            with open(local_fallback, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}


def write_json(s3_key: str, data: dict, local_fallback: str = None):
    """Write a JSON file to S3 and optionally to local path."""
    body = json.dumps(data, indent=2, ensure_ascii=True).encode("utf-8")
    try:
        _client().put_object(Bucket=BUCKET, Key=s3_key, Body=body, ContentType="application/json")
        logger.info(f"[S3] Wrote {s3_key}")
    except Exception as e:
        logger.warning(f"[S3] Write failed for {s3_key}: {e}")
    # Always write local copy too
    if local_fallback:
        os.makedirs(os.path.dirname(local_fallback), exist_ok=True)
        with open(local_fallback, "w", encoding="utf-8") as f:
            f.write(body.decode("utf-8"))


# ── Claim status ──────────────────────────────────────────────────────────────

def load_claim_status() -> dict:
    local = os.path.join(_LOCAL_DATA, "claim_status.json")
    return read_json("claims/claim_status.json", local)


def save_claim_status(data: dict):
    local = os.path.join(_LOCAL_DATA, "claim_status.json")
    write_json("claims/claim_status.json", data, local)


def update_patient_claim(patient_id: str, entry: dict):
    """Merge a single patient entry into claim_status.json on S3."""
    all_claims = load_claim_status()
    all_claims[patient_id] = {**all_claims.get(patient_id, {}), **entry, "updated_at": datetime.now().isoformat()}
    save_claim_status(all_claims)


# ── Patients CSV ──────────────────────────────────────────────────────────────

def read_patients_csv() -> str:
    """Return patients1.csv content as string."""
    try:
        obj = _client().get_object(Bucket=BUCKET, Key="patients/patients1.csv")
        return obj["Body"].read().decode("utf-8")
    except Exception as e:
        logger.warning(f"[S3] patients CSV read failed: {e} — using local")
        local = os.path.join(_LOCAL_DATA, "patients1.csv")
        if os.path.exists(local):
            with open(local, "r", encoding="utf-8") as f:
                return f.read()
        return ""


# ── ERA files ─────────────────────────────────────────────────────────────────

def upload_era(patient_id: str, filename: str, content: bytes) -> str:
    """Upload an ERA file to S3. Returns the S3 key."""
    key = f"era/{patient_id}/{filename}"
    try:
        _client().put_object(Bucket=BUCKET, Key=key, Body=content, ContentType="text/plain")
        logger.info(f"[S3] ERA uploaded: {key}")
    except Exception as e:
        logger.warning(f"[S3] ERA upload failed: {e}")
    return key


def get_era_url(s3_key: str, expires: int = 3600) -> str:
    """Generate a presigned URL for an ERA file."""
    try:
        url = _client().generate_presigned_url(
            "get_object", Params={"Bucket": BUCKET, "Key": s3_key}, ExpiresIn=expires
        )
        return url
    except Exception as e:
        logger.warning(f"[S3] Presigned URL failed: {e}")
        return ""


# ── Appeal PDFs ───────────────────────────────────────────────────────────────

def upload_appeal(patient_id: str, claim_id: str, pdf_bytes: bytes) -> str:
    """Upload an appeal PDF to S3. Returns the S3 key."""
    filename = f"appeal_{claim_id}_{patient_id}.pdf"
    key = f"appeals/{filename}"
    try:
        _client().put_object(Bucket=BUCKET, Key=key, Body=pdf_bytes, ContentType="application/pdf")
        logger.info(f"[S3] Appeal uploaded: {key}")
    except Exception as e:
        logger.warning(f"[S3] Appeal upload failed: {e}")
    return key


def list_appeals(patient_id: str = None) -> list:
    """List appeal files in S3."""
    try:
        prefix = f"appeals/appeal_{patient_id}" if patient_id else "appeals/"
        resp = _client().list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        return [obj["Key"] for obj in resp.get("Contents", [])]
    except Exception as e:
        logger.warning(f"[S3] List appeals failed: {e}")
        return []


# ── X12 837P storage ──────────────────────────────────────────────────────────

def upload_x12(patient_id: str, claim_id: str, x12_text: str):
    """Store X12 837P transaction in S3."""
    key = f"claims/x12/{claim_id}_{patient_id}.837"
    try:
        _client().put_object(Bucket=BUCKET, Key=key, Body=x12_text.encode("utf-8"), ContentType="text/plain")
        logger.info(f"[S3] X12 uploaded: {key}")
    except Exception as e:
        logger.warning(f"[S3] X12 upload failed: {e}")
    return key


# ── Logs ──────────────────────────────────────────────────────────────────────

def append_log(log_key: str, entry: dict):
    """Append a JSONL log entry to S3 (read-modify-write)."""
    try:
        try:
            obj = _client().get_object(Bucket=BUCKET, Key=log_key)
            existing = obj["Body"].read().decode("utf-8")
        except Exception:
            existing = ""
        line = json.dumps(entry, ensure_ascii=True) + "\n"
        _client().put_object(Bucket=BUCKET, Key=log_key, Body=(existing + line).encode("utf-8"), ContentType="application/x-ndjson")
    except Exception as e:
        logger.warning(f"[S3] Log append failed for {log_key}: {e}")
