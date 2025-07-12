from langgraph.graph import StateGraph, END
import sys
import os
try:
    from graph.nodes import (
        risk_predictor_node,
        auto_corrector_node,
        claim_submitter_node,
        appeal_generator_node,
        resubmitter_node,
        feedback_learner_node
    )
    from orchestrator import orchestrator
    from config.settings import Settings
    from tools.logger import secure_log
except ModuleNotFoundError:
    # If running from graph/ directory, add parent to sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nodes import (
        risk_predictor_node,
        auto_corrector_node,
        claim_submitter_node,
        appeal_generator_node,
        resubmitter_node,
        feedback_learner_node
    )
    from orchestrator import orchestrator
    from config.settings import Settings
    from tools.logger import secure_log
import asyncio
import time
from typing import TypedDict, Optional, List, Dict, Any
from tools.logger import secure_log

# Define state structure
class ClaimState(TypedDict):
    claim_id: str
    raw_data: dict
    risk_score: Optional[float]
    issues: Optional[List[str]]
    corrected_data: Optional[dict]
    submission_result: Optional[dict]
    appeal_packet: Optional[str]
    final_status: Optional[str]
    log: List[str]

class ClaimFlow:
    """Enhanced agentic claim processing workflow"""
    
    def __init__(self):
        self.settings = Settings()
        # Use secure_log instead of setup_logger
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        graph = StateGraph(ClaimState)

        # Add nodes
        graph.add_node("risk_predictor", risk_predictor_node)
        graph.add_node("auto_corrector", auto_corrector_node)
        graph.add_node("claim_submitter", claim_submitter_node)
        graph.add_node("appeal_generator", appeal_generator_node)
        graph.add_node("resubmitter", resubmitter_node)
        graph.add_node("feedback_learner", feedback_learner_node)

        # Entry point
        graph.set_entry_point("risk_predictor")

        # Define branching logic
        def risk_branch(state: ClaimState) -> str:
            risk_score = state.get("risk_score", 0)
            issues = state.get("issues", [])
            
            # Route to auto_corrector if:
            # 1. High risk score, OR
            # 2. Data quality issues detected (missing/incomplete data)
            needs_correction = False
            correction_reason = ""
            
            if risk_score and risk_score > self.settings.RISK_THRESHOLD:
                needs_correction = True
                correction_reason = f"High risk score ({risk_score})"
            
            # Check for data quality issues that need correction
            if issues:
                data_quality_issues = [
                    issue for issue in issues 
                    if any(keyword in issue.lower() for keyword in [
                        'missing', 'incomplete', 'no prior authorization', 
                        'unknown', 'lack of', 'absent'
                    ])
                ]
                
                if data_quality_issues:
                    needs_correction = True
                    if correction_reason:
                        correction_reason += f" and {len(data_quality_issues)} data quality issues"
                    else:
                        correction_reason = f"{len(data_quality_issues)} data quality issues detected"
            
            if needs_correction:
                state["log"].append(f"{correction_reason}, routing to auto_corrector")
                return "auto_corrector"
            else:
                state["log"].append(f"Low risk score ({risk_score}), no data issues, routing to claim_submitter")
                return "claim_submitter"

        def submit_branch(state: ClaimState) -> str:
            result = state.get("submission_result", {})
            status = result.get("status", "unknown")
            
            if status == "rejected" or status == "denied":
                state["log"].append(f"Claim {status}, routing to appeal_generator")
                return "appeal_generator"
            elif status == "approved":
                state["log"].append("Claim approved, routing to feedback_learner")
                return "feedback_learner"
            elif status == "pending":
                state["log"].append("Claim pending, routing to feedback_learner")
                return "feedback_learner"
            else:
                state["log"].append(f"Unknown status {status}, routing to feedback_learner")
                return "feedback_learner"

        def resubmit_branch(state: ClaimState) -> str:
            state["log"].append("Resubmission complete, routing to feedback_learner")
            return "feedback_learner"

        # Conditional routes
        graph.add_conditional_edges("risk_predictor", risk_branch)
        graph.add_edge("auto_corrector", "claim_submitter")
        graph.add_conditional_edges("claim_submitter", submit_branch)
        graph.add_edge("appeal_generator", "resubmitter")
        graph.add_edge("resubmitter", "feedback_learner")
        graph.add_edge("feedback_learner", END)

        return graph.compile()

    async def process_claim(self, claim_data: dict) -> dict:
        """Process a single claim through the agentic workflow"""
        claim_id = claim_data.get("claim_id", "unknown")
        try:
            secure_log("claim_flow", {"action": "start_processing", "claim_id": claim_id})

            # Create initial state with reasonable defaults
            initial_state: ClaimState = {
                "claim_id": claim_id,
                "raw_data": claim_data,
                "risk_score": 0.0,  # Default to 0.0, not None
                "issues": [],        # Always a list
                "corrected_data": None,
                "submission_result": None,
                "appeal_packet": None,
                "final_status": None,
                "log": [f"Processing started at {time.strftime('%Y-%m-%d %H:%M:%S')}"]
            }

            # Run the workflow
            final_state = await self.graph.ainvoke(initial_state)

            # Always provide non-null, reasonable values in result
            result = {
                "claim_id": claim_id,
                "status": final_state.get("final_status", "unknown"),
                "risk_score": final_state.get("risk_score", 0.0) if final_state.get("risk_score") is not None else 0.0,
                "issues": final_state.get("issues") if final_state.get("issues") is not None else [],
                "submission_result": final_state.get("submission_result"),
                "workflow_log": final_state.get("log", []),
                "processing_time": time.time(),
                "success": True
            }

            secure_log("claim_flow", {"action": "processing_complete", "claim_id": claim_id, "status": result['status']})
            return result

        except Exception as e:
            # Clean the error message to avoid Unicode issues
            clean_error = clean_unicode_for_json(str(e))
            secure_log("claim_flow", {"action": "processing_error", "claim_id": claim_id, "error": clean_error})
            return {
                "claim_id": claim_id,
                "status": "error",
                "error": clean_error,
                "success": False,
                "processing_time": time.time()
            }

    async def process_claims_batch(self, claims: List[dict]) -> List[dict]:
        """Process multiple claims concurrently"""
        secure_log("claim_flow", {"action": "batch_processing_start", "count": len(claims)})
        
        # Create tasks for concurrent processing
        tasks = [self.process_claim(claim) for claim in claims]
        
        # Wait for all claims to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "claim_id": claims[i].get("claim_id", f"claim_{i}"),
                    "status": "error",
                    "error": str(result),
                    "success": False
                })
            else:
                processed_results.append(result)
        
        return processed_results

# Legacy compatibility
def build_graph() -> StateGraph:
    """Legacy function for backward compatibility"""
    flow = ClaimFlow()
    return flow.graph

if __name__ == "__main__":
    import asyncio
    import json
    
    def clean_unicode_for_json(obj):
        """Clean Unicode characters that might cause encoding issues"""
        if isinstance(obj, dict):
            return {k: clean_unicode_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_unicode_for_json(item) for item in obj]
        elif isinstance(obj, str):
            # Replace problematic Unicode characters first
            result = obj.replace('\u2013', '-').replace('\u2014', '--').replace('\u2018', "'").replace('\u2019', "'").replace('\u201C', '"').replace('\u201D', '"').replace('\u2026', '...').replace('\u2713', 'v').replace('\u26a0', '!').replace('\u274c', 'X').replace('\u2705', 'v').replace('\ud83e\udde0', 'BRAIN').replace('\u2192', '->').replace('\u2190', '<-')
            # Remove any remaining non-ASCII characters as a final fallback
            result = ''.join(char if ord(char) < 128 else '?' for char in result)
            return result
        else:
            return obj
    
    print("[Standalone Test] Running ClaimFlow from graph/claim_flow.py\n")
    test_claim = {
        "claim_id": "DIRECT001",
        "patient_id": "PAT001",  # Use valid CSV patient ID
        "diagnosis": "Acute bronchitis", 
        "procedure": "Office visit",
        "amount": 123.45,
        "claim_amount": 123.45,
        "provider": "Test Provider",
        "date": "2025-07-12",
        "insurance_company": "BlueCross",
        "cpt_code": "99213",
        "icd_code": "J20.9",
        "prior_auth": "AUTH12345",  # Provide proper auth
        "medical_history": "Patient has history of respiratory infections",  # Provide history
        "provider_name": "Dr. Smith Medical Center",  # Provide provider name
        "patient_name": "John Doe",  # Provide patient name
        "age": 35,  # Provide age
        "gender": "Male",  # Provide gender
        "provider_npi": "1234567890"
    }
    flow = ClaimFlow()
    result = asyncio.run(flow.process_claim(test_claim))
    
    # Clean the result before JSON serialization
    clean_result = clean_unicode_for_json(result)
    
    # Bypass JSON for now and just show key results
    print("=" * 80)
    print("🏥 AGENTIC CLAIMS AI SYSTEM - PROCESSING RESULTS")
    print("=" * 80)
    
    # Stage-by-stage breakdown
    print(f"📋 CLAIM OVERVIEW:")
    print(f"   Claim ID: {clean_result.get('claim_id', 'unknown')}")
    print(f"   Final Status: {clean_result.get('status', 'unknown')}")
    print(f"   Risk Score: {clean_result.get('risk_score', 0.0):.2f}")
    print(f"   Success: {'✅ YES' if clean_result.get('success', False) else '❌ NO'}")
    print(f"   Processing Time: {clean_result.get('processing_time', 0.0):.2f}s")
    print()
    
    # Issues analysis
    issues = clean_result.get('issues', [])
    print(f"🔍 ISSUES ANALYSIS:")
    print(f"   Total Issues Found: {len(issues)}")
    if issues:
        for i, issue in enumerate(issues[:3], 1):  # Show first 3 issues
            print(f"   {i}. {issue[:60]}...")
        if len(issues) > 3:
            print(f"   ... and {len(issues) - 3} more issues")
    else:
        print("   ✅ No issues detected")
    print()
    
    # Submission result details
    if clean_result.get('submission_result'):
        sub_result = clean_result['submission_result']
        print(f"📤 SUBMISSION DETAILS:")
        print(f"   Submission Status: {sub_result.get('status', 'unknown')}")
        print(f"   Data Quality Score: {sub_result.get('data_quality_score', 0.0)}%")
        print(f"   Processed By: {sub_result.get('processed_by', 'Unknown')}")
        if sub_result.get('reason'):
            print(f"   Reason: {sub_result.get('reason', '')[:80]}...")
        if sub_result.get('approval_amount'):
            print(f"   Approval Amount: ${sub_result.get('approval_amount', 0.0):.2f}")
        print()
    
    # Workflow log summary
    workflow_log = clean_result.get('workflow_log', [])
    print(f"📊 WORKFLOW SUMMARY:")
    print(f"   Total Steps: {len(workflow_log)}")
    if workflow_log:
        print("   All Steps:")
        for i, step in enumerate(workflow_log, 1):  # Show ALL steps
            step_clean = str(step)[:100] + "..." if len(str(step)) > 100 else str(step)
            print(f"   {i}. {step_clean}")
    print()
    
    print("=" * 80)
    print("🤖 AGENTIC SYSTEM PERFORMANCE METRICS:")
    print("=" * 80)
    
    # Calculate performance metrics
    risk_level = "LOW" if clean_result.get('risk_score', 0.0) < 0.3 else "MEDIUM" if clean_result.get('risk_score', 0.0) < 0.7 else "HIGH"
    issue_count = len(clean_result.get('issues', []))
    
    print(f"🎯 Risk Assessment: {risk_level} ({clean_result.get('risk_score', 0.0):.2f})")
    print(f"🔧 Issues Detected: {issue_count}")
    print(f"⚡ Processing Speed: {'FAST' if clean_result.get('processing_time', 10) < 5 else 'NORMAL'}")
    
    if clean_result.get('submission_result'):
        quality_score = clean_result['submission_result'].get('data_quality_score', 0.0)
        quality_level = "EXCELLENT" if quality_score >= 90 else "GOOD" if quality_score >= 75 else "NEEDS_IMPROVEMENT"
        print(f"📈 Data Quality: {quality_level} ({quality_score}%)")
    
    print(f"🏆 Overall Performance: {'OPTIMAL' if clean_result.get('success', False) and issue_count <= 3 and clean_result.get('status') != 'error' else 'SUBOPTIMAL'}")
    print("=" * 80)
    
    # Try minimal JSON output
    try:
        minimal_result = {
            "claim_id": str(clean_result.get('claim_id', 'unknown')),
            "status": str(clean_result.get('status', 'unknown')),
            "success": bool(clean_result.get('success', False)),
            "risk_score": float(clean_result.get('risk_score', 0.0)),
            "issues_count": len(clean_result.get('issues', []))
        }
        print("MINIMAL JSON OUTPUT:")
        print(json.dumps(minimal_result, indent=2, ensure_ascii=True))
    except Exception as e:
        print(f"Even minimal JSON failed: {e}")
        print("Raw key values:")
        print(f"  claim_id: {repr(clean_result.get('claim_id'))}")
        print(f"  status: {repr(clean_result.get('status'))}")
        print(f"  success: {repr(clean_result.get('success'))}")
# LangGraph runtime
compiled_graph = build_graph()

async def run_claim_workflow(claim_input: dict) -> dict:
    """Legacy function for backward compatibility"""
    flow = ClaimFlow()
    return await flow.process_claim(claim_input)

async def run_agentic_claim_workflow(claim_input: dict) -> dict:
    """
    Run claim through advanced agentic orchestrator with MCP integration
    
    This is the hackathon showcase function that demonstrates:
    - Multi-agent coordination
    - MCP data source integration
    - Real-time decision making
    - Continuous learning
    """
    try:
        # Initialize orchestrator if not already done
        if not orchestrator.mcp_connected:
            await orchestrator.initialize()
        
        # Process claim through agentic orchestrator
        result = await orchestrator.process_claim_agentically(claim_input)
        
        return {
            "claim_id": claim_input.get("claim_id"),
            "status": "success",
            "agentic_result": result,
            "processing_type": "agentic_mcp",
            "timestamp": time.time()
        }
        
    except Exception as e:
        # Fallback to traditional workflow
        return await run_claim_workflow(claim_input)
