"""
Agent Orchestrator for Healthcare Claims Processing
Coordinates multiple AI agents and manages workflow execution
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.mcp_client import mcp_client
from tools.logger import secure_log

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """Orchestrates multiple AI agents for claim processing"""
    
    def __init__(self):
        self.mcp_client = mcp_client
        self.active_sessions = {}
        self.agent_status = {}
    
    async def initialize(self):
        """Initialize the orchestrator"""
        try:
            # Connect to MCP server
            await self.mcp_client.connect()
            
            # Get available tools
            tools = await self.mcp_client.get_available_tools()
            secure_log(f"Available MCP tools: {tools}", "INFO")
            
            # Initialize agent status
            self.agent_status = {
                "risk_predictor": "ready",
                "auto_corrector": "ready",
                "claim_submitter": "ready",
                "appeal_generator": "ready",
                "resubmitter": "ready",
                "feedback_learner": "ready"
            }
            
            logger.info("Agent orchestrator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            return False
    
    async def process_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a claim through the multi-agent system"""
        session_id = f"claim_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        try:
            # Store session
            self.active_sessions[session_id] = {
                "claim_data": claim_data,
                "started_at": datetime.now().isoformat(),
                "status": "processing"
            }
            
            # Step 1: Risk Assessment
            secure_log(f"Starting risk assessment for claim {session_id}", "INFO")
            risk_result = await self.mcp_client.get_risk_score(claim_data)
            
            if not risk_result:
                raise Exception("Risk assessment failed")
            
            # Step 2: Data Validation and Correction
            secure_log(f"Validating claim data for {session_id}", "INFO")
            validation_result = await self.mcp_client.validate_claim(claim_data)
            
            if not validation_result or not validation_result.get("is_valid"):
                # Auto-correct if possible
                corrected_data = await self.auto_correct_claim(claim_data, validation_result)
                if corrected_data:
                    claim_data = corrected_data
                else:
                    raise Exception("Claim validation failed and auto-correction not possible")
            
            # Step 3: Format claim data
            secure_log(f"Formatting claim data for {session_id}", "INFO")
            formatted_result = await self.mcp_client.format_claim_data(claim_data)
            
            if not formatted_result:
                raise Exception("Claim formatting failed")
            
            formatted_data = formatted_result.get("formatted_data")
            
            # Step 4: Submit to insurer
            secure_log(f"Submitting claim to insurer for {session_id}", "INFO")
            submission_result = await self.mcp_client.submit_to_insurer(formatted_data)
            
            if not submission_result:
                raise Exception("Claim submission failed")
            
            # Step 5: Handle result
            final_result = {
                "session_id": session_id,
                "claim_id": submission_result.get("claim_id"),
                "status": submission_result.get("status"),
                "risk_score": risk_result.get("risk_score"),
                "risk_level": risk_result.get("risk_level"),
                "submitted_at": submission_result.get("submitted_at"),
                "processing_time_days": submission_result.get("estimated_processing_days"),
                "success": True
            }
            
            # Update session
            self.active_sessions[session_id]["status"] = "completed"
            self.active_sessions[session_id]["result"] = final_result
            
            secure_log(f"Claim processing completed for {session_id}", "INFO")
            return final_result
            
        except Exception as e:
            error_result = {
                "session_id": session_id,
                "success": False,
                "error": str(e),
                "failed_at": datetime.now().isoformat()
            }
            
            # Update session
            self.active_sessions[session_id]["status"] = "failed"
            self.active_sessions[session_id]["error"] = str(e)
            
            secure_log(f"Claim processing failed for {session_id}: {e}", "ERROR")
            return error_result
    
    async def auto_correct_claim(self, claim_data: Dict[str, Any], validation_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Auto-correct claim data based on validation results"""
        try:
            corrected_data = claim_data.copy()
            missing_fields = validation_result.get("missing_fields", [])
            
            # Basic auto-correction logic
            for field in missing_fields:
                if field == "service_date" and not corrected_data.get("service_date"):
                    corrected_data["service_date"] = datetime.now().strftime("%Y-%m-%d")
                elif field == "provider_id" and not corrected_data.get("provider_id"):
                    corrected_data["provider_id"] = "PROV001"  # Default provider
                elif field == "claim_amount" and not corrected_data.get("claim_amount"):
                    corrected_data["claim_amount"] = 100.0  # Default minimal amount
            
            # Re-validate
            validation_result = await self.mcp_client.validate_claim(corrected_data)
            
            if validation_result and validation_result.get("is_valid"):
                secure_log("Auto-correction successful", "INFO")
                return corrected_data
            else:
                secure_log("Auto-correction failed", "WARNING")
                return None
                
        except Exception as e:
            secure_log(f"Auto-correction error: {e}", "ERROR")
            return None
    
    async def handle_denial(self, denial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle claim denial by generating appeal"""
        try:
            # Generate appeal
            appeal_result = await self.mcp_client.generate_appeal(denial_data)
            
            if not appeal_result:
                raise Exception("Appeal generation failed")
            
            # Update denial patterns for learning
            await self.mcp_client.update_denial_patterns(denial_data)
            
            return {
                "appeal_generated": True,
                "appeal_id": appeal_result.get("appeal_id"),
                "appeal_text": appeal_result.get("appeal_text"),
                "status": "ready_for_submission"
            }
            
        except Exception as e:
            secure_log(f"Denial handling error: {e}", "ERROR")
            return {
                "appeal_generated": False,
                "error": str(e)
            }
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a processing session"""
        return self.active_sessions.get(session_id)
    
    async def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get all active sessions"""
        return list(self.active_sessions.values())
    
    async def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old sessions"""
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, session_data in self.active_sessions.items():
            started_at = datetime.fromisoformat(session_data["started_at"])
            age_hours = (current_time - started_at).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
        
        if sessions_to_remove:
            secure_log(f"Cleaned up {len(sessions_to_remove)} old sessions", "INFO")
    
    async def shutdown(self):
        """Shutdown the orchestrator"""
        await self.mcp_client.disconnect()
        logger.info("Agent orchestrator shut down")

# Global orchestrator instance
orchestrator = AgentOrchestrator()
