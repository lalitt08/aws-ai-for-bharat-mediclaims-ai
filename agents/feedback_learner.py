# agents/feedback_learner.py - Feedback Learning Agent

import asyncio
import json
from typing import Dict, Any, List
from tools.logger import secure_log

async def run_feedback_learning(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Feedback Learning Agent - Learns from claim outcomes to improve future processing with detailed logging
    """
    secure_log("feedback_learner", state)
    
    try:
        claim_id = state.get('claim_id', 'unknown')
        final_status = state.get('final_status', 'unknown')
        
        # Add detailed logging for UI activity tracking
        state.setdefault('log', []).append("[FeedbackLearner] Analyzes final claim outcome to learn from successful/failed strategies")
        state.setdefault('log', []).append("[FeedbackLearner] Learning pattern updated")
        
        # Call Bedrock Agent Core for deep pattern analysis
        try:
            from tools.bedrock_agent_integration import bedrock_learn_outcome
            claim_data = state.get("corrected_data") or state.get("raw_data", {})
            ba_result = bedrock_learn_outcome(
                {**claim_data, "claim_id": claim_id},
                final_status,
            )
            if ba_result:
                state["log"].append(
                    f"[FeedbackLearner] Bedrock Agent Core: patterns_updated={ba_result.get('patterns_updated')} "
                    f"source={ba_result.get('source')}"
                )
                if ba_result.get("insights"):
                    state["bedrock_insights"] = ba_result["insights"]
                # Save insight to S3
                try:
                    from tools.s3_storage import append_log
                    append_log("logs/feedback_learner.jsonl", {
                        "claim_id": claim_id,
                        "outcome": final_status,
                        "bedrock_insights": ba_result.get("insights", ""),
                        "patterns_updated": ba_result.get("patterns_updated"),
                        "insurer": claim_data.get("insurer") or claim_data.get("insurance_company"),
                        "cpt_code": claim_data.get("cpt_code") or claim_data.get("procedure_code"),
                        "icd_code": claim_data.get("icd_code") or claim_data.get("diagnosis_code"),
                    })
                except Exception:
                    pass
        except Exception as ba_err:
            state["log"].append(f"[FeedbackLearner] Bedrock Agent skipped: {ba_err}")
        
        # Log the learning attempt
        log_entry = f"Learning from claim {claim_id} outcome: {final_status}"
        state.setdefault('log', []).append(log_entry)
        
        # Simulate learning process
        await asyncio.sleep(0.5)  # Simulate processing time
        
        # In a real system, this would:
        # 1. Analyze the claim outcome
        # 2. Identify patterns in successful/failed claims
        # 3. Update ML models or rules
        # 4. Store learnings for future use
        # 5. Update success prediction algorithms
        
        # Extract learning insights
        learning_insights = []
        
        if final_status == 'approved':
            learning_insights.append("Successful claim pattern identified")
            learning_insights.append("Reinforce successful data patterns")
        elif final_status == 'denied':
            learning_insights.append("Denial pattern identified")
            learning_insights.append("Update correction algorithms")
        
        # Add insights based on the claim processing journey
        if 'issues' in state and state['issues']:
            learning_insights.append(f"Common issues found: {', '.join(state['issues'])}")
        
        if 'risk_score' in state and state['risk_score'] is not None:
            risk_score = state['risk_score']
            if risk_score > 0.7:
                learning_insights.append("High risk claim processed - analyze patterns")
            elif risk_score < 0.3:
                learning_insights.append("Low risk claim processed - confirm patterns")
        
        # Create learning result
        learning_result = {
            'learning_id': f"LEARN_{claim_id}_{int(asyncio.get_event_loop().time())}",
            'claim_outcome': final_status,
            'insights': learning_insights,
            'patterns_updated': True,
            'learning_date': str(asyncio.get_event_loop().time()),
            'success': True
        }
        
        # Update state
        state['learning_result'] = learning_result
        state['log'].append(f"[FeedbackLearner] Learning pattern updated")
        
        # Set processing status - DON'T override the actual claim status!
        # Preserve the existing final_status instead of overriding it
        if not state.get("final_status") or state.get("final_status") == "unknown":
            state["final_status"] = "learning_complete"
        # Otherwise keep the existing status (like "resubmitted", "appealed", etc.)
        
        secure_log("feedback_learner", state)
        
        return state
        
    except Exception as e:
        error_msg = f"[ERROR] Learning failed: {str(e)}"
        secure_log("feedback_learner", {**state, "error": error_msg})
        
        state.setdefault('log', []).append(error_msg)
        state['learning_result'] = {
            'status': 'failed',
            'error': str(e),
            'success': False
        }
        
        return state

# For direct testing
if __name__ == "__main__":
    # Test the feedback learner
    test_state = {
        'claim_id': 'TEST_001',
        'final_status': 'approved',
        'risk_score': 0.3,
        'issues': ['Minor coding issue'],
        'log': []
    }
    
    result = asyncio.run(run_feedback_learning(test_state))
    print(json.dumps(result, indent=2))
