"""
Claim Status Manager
===================

Manages persistent storage and retrieval of claim processing status.
This bridges the gap between the claim processing workflow and the dashboard.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ClaimStatusManager:
    """Manages claim status persistence"""
    
    def __init__(self, status_file_path: str = None):
        if status_file_path is None:
            # Default to data directory
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            os.makedirs(data_dir, exist_ok=True)
            status_file_path = os.path.join(data_dir, 'claim_status.json')
        
        self.status_file_path = status_file_path
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Ensure the status file exists"""
        try:
            if not os.path.exists(self.status_file_path):
                with open(self.status_file_path, 'w') as f:
                    json.dump({}, f)
        except Exception as e:
            logger.error(f"Error creating status file: {e}")
    
    def save_claim_status(self, claim_id: str, patient_id: str, status: str, additional_data: Dict[str, Any] = None):
        """Save claim status to persistent storage"""
        try:
            # Load existing statuses
            statuses = self.load_all_statuses()
            
            # Create status entry
            status_entry = {
                "claim_id": claim_id,
                "patient_id": patient_id,
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Add additional data if provided
            if additional_data:
                status_entry.update(additional_data)
            
            # Save status (use patient_id as key for easy lookup)
            statuses[patient_id] = status_entry
            
            # Write back to file
            with open(self.status_file_path, 'w') as f:
                json.dump(statuses, f, indent=2)
            
            logger.info(f"Saved claim status for patient {patient_id}: {status}")
            
        except Exception as e:
            logger.error(f"Error saving claim status: {e}")
    
    def get_claim_status(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get claim status for a specific patient"""
        try:
            statuses = self.load_all_statuses()
            return statuses.get(patient_id)
        except Exception as e:
            logger.error(f"Error getting claim status for {patient_id}: {e}")
            return None
    
    def load_all_statuses(self) -> Dict[str, Any]:
        """Load all claim statuses from file"""
        try:
            if os.path.exists(self.status_file_path):
                with open(self.status_file_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading claim statuses: {e}")
            return {}
    
    def update_patient_status(self, patient_id: str, new_status: str):
        """Update just the status for a patient"""
        try:
            statuses = self.load_all_statuses()
            if patient_id in statuses:
                statuses[patient_id]["status"] = new_status
                statuses[patient_id]["updated_at"] = datetime.now().isoformat()
                
                with open(self.status_file_path, 'w') as f:
                    json.dump(statuses, f, indent=2)
                
                logger.info(f"Updated status for patient {patient_id}: {new_status}")
            else:
                logger.warning(f"No existing status found for patient {patient_id}")
        except Exception as e:
            logger.error(f"Error updating patient status: {e}")
    
    def get_patients_by_status(self, status: str) -> list:
        """Get all patients with a specific status"""
        try:
            statuses = self.load_all_statuses()
            return [entry for entry in statuses.values() if entry.get("status") == status]
        except Exception as e:
            logger.error(f"Error getting patients by status: {e}")
            return []
    
    def clear_all_statuses(self):
        """Clear all claim statuses (for testing)"""
        try:
            with open(self.status_file_path, 'w') as f:
                json.dump({}, f)
            logger.info("Cleared all claim statuses")
        except Exception as e:
            logger.error(f"Error clearing claim statuses: {e}")


# Global instance for easy access
claim_status_manager = ClaimStatusManager()

# Convenience functions
def save_claim_status(claim_id: str, patient_id: str, status: str, additional_data: Dict[str, Any] = None):
    """Save claim status - convenience function"""
    return claim_status_manager.save_claim_status(claim_id, patient_id, status, additional_data)

def get_claim_status(patient_id: str) -> Optional[Dict[str, Any]]:
    """Get claim status - convenience function"""
    return claim_status_manager.get_claim_status(patient_id)

def update_patient_status(patient_id: str, new_status: str):
    """Update patient status - convenience function"""
    return claim_status_manager.update_patient_status(patient_id, new_status)
