"""
Claim Status Manager — S3-backed persistent storage.
Primary: S3 bucket (alpha-claims-demo-390783052961/claims/claim_status.json)
Fallback: local data/claim_status.json
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Try S3 storage first
try:
    from tools.s3_storage import load_claim_status as _s3_load, save_claim_status as _s3_save
    _HAS_S3 = True
except ImportError:
    _HAS_S3 = False

_LOCAL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "claim_status.json")


def _load_all() -> dict:
    if _HAS_S3:
        return _s3_load()
    if os.path.exists(_LOCAL_PATH):
        with open(_LOCAL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_all(data: dict):
    if _HAS_S3:
        _s3_save(data)
        return
    os.makedirs(os.path.dirname(_LOCAL_PATH), exist_ok=True)
    with open(_LOCAL_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


class ClaimStatusManager:
    """Manages claim status persistence — S3 primary, local fallback."""

    def __init__(self, status_file_path: str = None):
        # status_file_path kept for backward compat but ignored (S3 used)
        self.status_file_path = status_file_path or _LOCAL_PATH
        if not _HAS_S3:
            os.makedirs(os.path.dirname(self.status_file_path), exist_ok=True)
            if not os.path.exists(self.status_file_path):
                with open(self.status_file_path, "w") as f:
                    json.dump({}, f)

    def save_claim_status(self, claim_id: str, patient_id: str, status: str, additional_data: Dict[str, Any] = None):
        statuses = _load_all()
        entry = {
            "claim_id": claim_id,
            "patient_id": patient_id,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        if additional_data:
            entry.update(additional_data)
        statuses[patient_id] = entry
        _save_all(statuses)

        # Also upload X12 to S3 if present
        if _HAS_S3 and additional_data and additional_data.get("x12_837p"):
            try:
                from tools.s3_storage import upload_x12
                upload_x12(patient_id, claim_id, additional_data["x12_837p"])
            except Exception as e:
                logger.warning(f"X12 S3 upload failed: {e}")

        logger.info(f"Saved claim status for {patient_id}: {status}")

    def get_claim_status(self, patient_id: str) -> Optional[Dict[str, Any]]:
        return _load_all().get(patient_id)

    def load_all_statuses(self) -> Dict[str, Any]:
        return _load_all()

    def update_patient_status(self, patient_id: str, new_status: str):
        statuses = _load_all()
        if patient_id in statuses:
            statuses[patient_id]["status"] = new_status
            statuses[patient_id]["updated_at"] = datetime.now().isoformat()
            _save_all(statuses)

    def get_patients_by_status(self, status: str) -> list:
        return [e for e in _load_all().values() if e.get("status") == status]

    def clear_all_statuses(self):
        _save_all({})


# Global instance
claim_status_manager = ClaimStatusManager()

def save_claim_status(claim_id, patient_id, status, additional_data=None):
    return claim_status_manager.save_claim_status(claim_id, patient_id, status, additional_data)

def get_claim_status(patient_id):
    return claim_status_manager.get_claim_status(patient_id)

def update_patient_status(patient_id, new_status):
    return claim_status_manager.update_patient_status(patient_id, new_status)
