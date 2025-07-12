"""
Unified Data Access Layer for Healthcare Claims System
======================================================

This module provides a unified interface for data access that automatically
selects between standalone (direct CSV) and MCP (tool-based) modes.
"""

import asyncio
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.csv_data_loader import PatientLoader, DenialLearningLoader
from tools.logger import secure_log

logger = logging.getLogger(__name__)

class UnifiedDataAccess:
    """
    Unified data access interface that adapts to operational mode
    
    Supports two modes:
    - 'standalone': Direct CSV loading via PatientLoader
    - 'mcp': Tool-based access via MCP client
    """
    
    def __init__(self, settings):
        self.mode = settings.OPERATIONAL_MODE
        self.settings = settings
        
        # Initialize based on mode
        if self.mode == 'standalone':
            self._init_standalone()
        elif self.mode == 'mcp':
            self._init_mcp()
        else:
            raise ValueError(f"Invalid operational mode: {self.mode}")
    
    def _init_standalone(self):
        """Initialize standalone mode with direct CSV access"""
        try:
            self.patient_loader = PatientLoader()
            self.denial_loader = DenialLearningLoader()
            secure_log("Initialized standalone data access", "INFO")
        except Exception as e:
            secure_log(f"Error initializing standalone mode: {e}", "ERROR")
            raise
    
    def _init_mcp(self):
        """Initialize MCP mode with client connection"""
        try:
            from orchestrator.mcp_client import mcp_client
            self.mcp_client = mcp_client
            secure_log("Initialized MCP data access", "INFO")
        except Exception as e:
            secure_log(f"Error initializing MCP mode: {e}", "ERROR")
            raise
    
    async def get_patient_data(self, patient_id: str) -> Dict[str, Any]:
        """Get patient data using appropriate access method"""
        if self.mode == 'standalone':
            return self._get_patient_data_standalone(patient_id)
        elif self.mode == 'mcp':
            return await self._get_patient_data_mcp(patient_id)
    
    def _get_patient_data_standalone(self, patient_id: str) -> Dict[str, Any]:
        """Get patient data directly from CSV"""
        try:
            patient = self.patient_loader.get_patient_by_id(patient_id)
            if not patient:
                raise ValueError(f"Patient {patient_id} not found")
            
            return {
                "patient_id": patient_id,
                "data": patient,
                "source": "standalone_csv",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            secure_log(f"Error getting patient data (standalone): {e}", "ERROR")
            raise
    
    async def _get_patient_data_mcp(self, patient_id: str) -> Dict[str, Any]:
        """Get patient data via MCP client"""
        try:
            result = await self.mcp_client.call_tool('get_patient_data', {
                'patient_id': patient_id
            })
            result['source'] = 'mcp_server'
            return result
        except Exception as e:
            secure_log(f"Error getting patient data (MCP): {e}", "ERROR")
            raise
    
    async def get_all_patients(self) -> List[Dict[str, Any]]:
        """Get all patients using appropriate access method"""
        if self.mode == 'standalone':
            return self._get_all_patients_standalone()
        elif self.mode == 'mcp':
            return await self._get_all_patients_mcp()
    
    def _get_all_patients_standalone(self) -> List[Dict[str, Any]]:
        """Get all patients directly from CSV"""
        try:
            patients = self.patient_loader.get_all_patients()
            return [{
                "data": patient,
                "source": "standalone_csv",
                "timestamp": datetime.now().isoformat()
            } for patient in patients]
        except Exception as e:
            secure_log(f"Error getting all patients (standalone): {e}", "ERROR")
            raise
    
    async def _get_all_patients_mcp(self) -> List[Dict[str, Any]]:
        """Get all patients via MCP client"""
        try:
            result = await self.mcp_client.call_tool('get_all_patients', {})
            if isinstance(result, dict) and 'patients' in result:
                return result['patients']
            return result
        except Exception as e:
            secure_log(f"Error getting all patients (MCP): {e}", "ERROR")
            raise
    
    async def validate_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate claim data using appropriate method"""
        if self.mode == 'standalone':
            return self._validate_claim_standalone(claim_data)
        elif self.mode == 'mcp':
            return await self._validate_claim_mcp(claim_data)
    
    def _validate_claim_standalone(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate claim data using local logic"""
        required_fields = [
            "patient_id", "procedure_code", "diagnosis_code", 
            "claim_amount", "service_date", "provider_id"
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in claim_data or not claim_data[field]:
                missing_fields.append(field)
        
        return {
            "is_valid": len(missing_fields) == 0,
            "missing_fields": missing_fields,
            "validation_timestamp": datetime.now().isoformat(),
            "source": "standalone_validation"
        }
    
    async def _validate_claim_mcp(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate claim data via MCP client"""
        try:
            result = await self.mcp_client.call_tool('validate_claim', {
                'claim_data': claim_data
            })
            result['source'] = 'mcp_validation'
            return result
        except Exception as e:
            secure_log(f"Error validating claim (MCP): {e}", "ERROR")
            raise
    
    async def submit_to_insurer(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit claim to insurer using appropriate method"""
        if self.mode == 'standalone':
            return self._submit_to_insurer_standalone(claim_data)
        elif self.mode == 'mcp':
            return await self._submit_to_insurer_mcp(claim_data)
    
    def _submit_to_insurer_standalone(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit claim using local insurer API simulation"""
        try:
            from tools.insurer_api import InsurerAPI
            api = InsurerAPI()
            return api.submit_claim(claim_data)
        except Exception as e:
            secure_log(f"Error submitting claim (standalone): {e}", "ERROR")
            # Return mock response for testing
            return {
                "status": "submitted",
                "claim_id": f"CLM_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "source": "standalone_submission"
            }
    
    async def _submit_to_insurer_mcp(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit claim via MCP client"""
        try:
            result = await self.mcp_client.call_tool('submit_to_insurer', {
                'claim_data': claim_data
            })
            result['source'] = 'mcp_submission'
            return result
        except Exception as e:
            secure_log(f"Error submitting claim (MCP): {e}", "ERROR")
            raise
    
    async def get_risk_score(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get risk score using appropriate method"""
        if self.mode == 'standalone':
            return self._get_risk_score_standalone(patient_data)
        elif self.mode == 'mcp':
            return await self._get_risk_score_mcp(patient_data)
    
    def _get_risk_score_standalone(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk score using local logic"""
        # Simple risk scoring logic
        risk_score = 0.5  # Default medium risk
        
        # Adjust based on patient data
        if 'age' in patient_data:
            age = int(patient_data['age'])
            if age > 65:
                risk_score += 0.2
            elif age < 18:
                risk_score += 0.1
        
        if 'chronic_conditions' in patient_data:
            conditions = patient_data['chronic_conditions']
            if conditions and len(conditions) > 0:
                risk_score += 0.3
        
        return {
            "risk_score": min(risk_score, 1.0),
            "risk_level": "high" if risk_score > 0.7 else "medium" if risk_score > 0.3 else "low",
            "timestamp": datetime.now().isoformat(),
            "source": "standalone_risk_assessment"
        }
    
    async def _get_risk_score_mcp(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get risk score via MCP client"""
        try:
            result = await self.mcp_client.call_tool('get_risk_score', {
                'patient_data': patient_data
            })
            result['source'] = 'mcp_risk_assessment'
            return result
        except Exception as e:
            secure_log(f"Error getting risk score (MCP): {e}", "ERROR")
            raise
    
    def get_mode_info(self) -> Dict[str, Any]:
        """Get information about current operational mode"""
        return {
            "mode": self.mode,
            "description": {
                "standalone": "Direct CSV access for local processing",
                "mcp": "Tool-based access via MCP server"
            }.get(self.mode, "Unknown mode"),
            "timestamp": datetime.now().isoformat()
        }


# Convenience function for creating unified data access
def create_data_access(settings=None):
    """Create unified data access instance with fallback to standalone mode"""
    if settings is None:
        # Create minimal settings for standalone mode
        class MinimalSettings:
            OPERATIONAL_MODE = 'standalone'
        settings = MinimalSettings()
    
    return UnifiedDataAccess(settings)
