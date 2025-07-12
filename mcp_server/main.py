"""
MCP (Model Context Protocol) Server for Healthcare Claims Processing
Provides external tool integration for the agentic system
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import json
import os
from datetime import datetime
import pandas as pd

# Import agentic system components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.csv_data_loader import PatientLoader, DenialLearningLoader
from tools.logger import secure_log
from config.settings import Settings

app = FastAPI(title="Healthcare Claims MCP Server", version="1.0.0")

# Initialize components
settings = Settings()
patient_loader = PatientLoader()
denial_loader = DenialLearningLoader()

class MCPRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    session_id: Optional[str] = None

class MCPResponse(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    session_id: Optional[str] = None

class MCPToolRegistry:
    """Registry of available MCP tools"""
    
    def __init__(self):
        self.tools = {
            "get_patient_data": self.get_patient_data,
            "get_all_patients": self.get_all_patients,
            "validate_claim": self.validate_claim,
            "submit_to_insurer": self.submit_to_insurer,
            "generate_appeal": self.generate_appeal,
            "update_denial_patterns": self.update_denial_patterns,
            "get_risk_score": self.get_risk_score,
            "format_claim_data": self.format_claim_data
        }
    
    async def get_patient_data(self, patient_id: str) -> Dict[str, Any]:
        """Get patient data from CSV"""
        try:
            patient = patient_loader.get_patient_by_id(patient_id)
            if not patient:
                raise ValueError(f"Patient {patient_id} not found")
            
            return {
                "patient_id": patient_id,
                "data": patient,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            secure_log(f"Error getting patient data: {e}", "ERROR")
            raise e
    
    async def get_all_patients(self) -> Dict[str, Any]:
        """Get all patients from CSV"""
        try:
            patients = patient_loader.get_all_patients()
            return {
                "patients": [{
                    "data": patient,
                    "source": "mcp_server",
                    "timestamp": datetime.now().isoformat()
                } for patient in patients],
                "total_count": len(patients),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            secure_log(f"Error getting all patients: {e}", "ERROR")
            raise e
    
    async def validate_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate claim data structure"""
        try:
            required_fields = [
                "patient_id", "procedure_code", "diagnosis_code", 
                "claim_amount", "service_date", "provider_id"
            ]
            
            missing_fields = []
            for field in required_fields:
                if field not in claim_data or not claim_data[field]:
                    missing_fields.append(field)
            
            validation_result = {
                "is_valid": len(missing_fields) == 0,
                "missing_fields": missing_fields,
                "validation_timestamp": datetime.now().isoformat()
            }
            
            if not validation_result["is_valid"]:
                secure_log(f"Claim validation failed: {missing_fields}", "WARNING")
            
            return validation_result
            
        except Exception as e:
            secure_log(f"Error validating claim: {e}", "ERROR")
            raise e
    
    async def submit_to_insurer(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit claim to insurance provider"""
        try:
            # Simulate insurance API call
            insurer = claim_data.get("insurer", "Unknown")
            claim_id = f"CLM-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Mock response based on claim amount for simulation
            claim_amount = float(claim_data.get("claim_amount", 0))
            
            if claim_amount > 5000:
                status = "pending_review"
            elif claim_amount > 1000:
                status = "approved"
            else:
                status = "approved"
            
            result = {
                "claim_id": claim_id,
                "status": status,
                "insurer": insurer,
                "submitted_at": datetime.now().isoformat(),
                "estimated_processing_days": 3 if status == "approved" else 7
            }
            
            secure_log(f"Claim submitted to {insurer}: {claim_id}", "INFO")
            return result
            
        except Exception as e:
            secure_log(f"Error submitting claim: {e}", "ERROR")
            raise e
    
    async def generate_appeal(self, denial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appeal letter for denied claim"""
        try:
            denial_reason = denial_data.get("denial_reason", "Unspecified")
            claim_id = denial_data.get("claim_id", "Unknown")
            
            # Generate appeal based on denial reason
            appeal_templates = {
                "medical_necessity": "This procedure was medically necessary based on patient symptoms and medical history.",
                "prior_authorization": "Prior authorization was obtained as required by the insurance policy.",
                "documentation": "All required documentation has been provided with this appeal.",
                "coding_error": "The procedure and diagnosis codes have been reviewed and are accurate."
            }
            
            appeal_text = appeal_templates.get(
                denial_reason.lower().replace(" ", "_"),
                f"We respectfully appeal the denial of claim {claim_id} and request reconsideration."
            )
            
            appeal_result = {
                "appeal_id": f"APP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "claim_id": claim_id,
                "appeal_text": appeal_text,
                "generated_at": datetime.now().isoformat(),
                "status": "ready_for_submission"
            }
            
            secure_log(f"Appeal generated for claim {claim_id}", "INFO")
            return appeal_result
            
        except Exception as e:
            secure_log(f"Error generating appeal: {e}", "ERROR")
            raise e
    
    async def update_denial_patterns(self, denial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update denial learning patterns"""
        try:
            # Add to denial learning dataset
            denial_record = {
                "denial_id": f"DEN-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "patient_id": denial_data.get("patient_id"),
                "denial_category": denial_data.get("denial_reason"),
                "denial_reason": denial_data.get("denial_reason"),
                "timestamp": datetime.now().isoformat(),
                "resolved": False
            }
            
            # This would normally update the CSV file
            secure_log(f"Denial pattern updated: {denial_record}", "INFO")
            
            return {
                "success": True,
                "denial_id": denial_record["denial_id"],
                "updated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            secure_log(f"Error updating denial patterns: {e}", "ERROR")
            raise e
    
    async def get_risk_score(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk score for claim"""
        try:
            # Simple risk scoring logic
            risk_factors = []
            risk_score = 0.0
            
            # Amount-based risk
            amount = float(claim_data.get("claim_amount", 0))
            if amount > 10000:
                risk_score += 0.3
                risk_factors.append("high_amount")
            elif amount > 5000:
                risk_score += 0.2
                risk_factors.append("medium_amount")
            
            # Procedure-based risk
            procedure = claim_data.get("procedure_code", "")
            if procedure.startswith("99"):
                risk_score += 0.1
                risk_factors.append("evaluation_procedure")
            
            # Patient history risk (mock)
            patient_id = claim_data.get("patient_id", "")
            if patient_id:
                # Mock: some patients have higher risk
                if int(patient_id.replace("PAT", "")) % 3 == 0:
                    risk_score += 0.2
                    risk_factors.append("patient_history")
            
            risk_result = {
                "risk_score": min(risk_score, 1.0),  # Cap at 1.0
                "risk_level": "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low",
                "risk_factors": risk_factors,
                "calculated_at": datetime.now().isoformat()
            }
            
            return risk_result
            
        except Exception as e:
            secure_log(f"Error calculating risk score: {e}", "ERROR")
            raise e
    
    async def format_claim_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format claim data for submission"""
        try:
            formatted_data = {
                "claim_header": {
                    "patient_id": raw_data.get("patient_id"),
                    "claim_id": raw_data.get("claim_id", f"CLM-{datetime.now().strftime('%Y%m%d%H%M%S')}"),
                    "submission_date": datetime.now().isoformat()
                },
                "patient_info": {
                    "name": raw_data.get("patient_name"),
                    "dob": raw_data.get("date_of_birth"),
                    "insurance_id": raw_data.get("insurance_id")
                },
                "service_details": {
                    "procedure_code": raw_data.get("procedure_code"),
                    "diagnosis_code": raw_data.get("diagnosis_code"),
                    "service_date": raw_data.get("service_date"),
                    "provider_id": raw_data.get("provider_id")
                },
                "financial": {
                    "claim_amount": float(raw_data.get("claim_amount", 0)),
                    "currency": "USD"
                }
            }
            
            return {
                "formatted_data": formatted_data,
                "formatted_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            secure_log(f"Error formatting claim data: {e}", "ERROR")
            raise e

# Initialize tool registry
tool_registry = MCPToolRegistry()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Healthcare Claims MCP Server",
        "version": "1.0.0",
        "status": "running",
        "available_tools": list(tool_registry.tools.keys())
    }

@app.get("/tools")
async def list_tools():
    """List available MCP tools"""
    return {
        "tools": list(tool_registry.tools.keys()),
        "total_tools": len(tool_registry.tools)
    }

@app.post("/execute", response_model=MCPResponse)
async def execute_tool(request: MCPRequest):
    """Execute an MCP tool"""
    try:
        if request.tool_name not in tool_registry.tools:
            raise HTTPException(
                status_code=400, 
                detail=f"Tool '{request.tool_name}' not found"
            )
        
        tool_func = tool_registry.tools[request.tool_name]
        result = await tool_func(**request.arguments)
        
        return MCPResponse(
            success=True,
            result=result,
            session_id=request.session_id
        )
        
    except Exception as e:
        secure_log(f"MCP tool execution error: {e}", "ERROR")
        return MCPResponse(
            success=False,
            error=str(e),
            session_id=request.session_id
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": "Healthcare Claims MCP Server"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
