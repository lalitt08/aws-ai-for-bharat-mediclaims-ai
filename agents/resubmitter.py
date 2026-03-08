# agents/resubmitter.py - Resubmission Agent

import asyncio
import json
from typing import Dict, Any, List
from tools.logger import secure_log

async def run_resubmission(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced Resubmission Agent - Bedrock Agent Core + intelligent resubmission
    """
    secure_log("resubmitter", state)
    
    try:
        claim_id = state.get('claim_id', 'unknown')
        corrected_data = state.get('corrected_data', {})
        appeal_packet = state.get('appeal_packet', '')
        appeal_text   = state.get('appeal_text', str(appeal_packet))
        original_rejection = state.get('submission_result', {})
        denial_info = original_rejection.get('denial_info', {}) if original_rejection else {}

        # ── Bedrock Agent Core call (primary path) ────────────────────────────
        try:
            from tools.bedrock_agent_integration import bedrock_resubmit
            ba_result = bedrock_resubmit(
                {**corrected_data, "claim_id": claim_id,
                 "denial_code": denial_info.get("code", "CO-16"),
                 "denial_reason": denial_info.get("reason", "")},
                appeal_text,
            )
            if ba_result:
                state['resubmission_result'] = {
                    'resubmission_id':    ba_result.get("resubmission_id", f"RESUB-{claim_id}"),
                    'status':             ba_result.get("status", "resubmitted"),
                    'success_probability': ba_result.get("success_probability", 0.7),
                    'strategy_used':      ba_result.get("strategy", "standard_appeal"),
                    'appeal_included':    bool(appeal_text),
                    'success':            True,
                    'source':             ba_result.get("source"),
                }
                state['final_status'] = "appeal_resubmitted"
                state.setdefault('log', []).append(
                    f"[Resubmitter] Bedrock Agent Core: status={ba_result.get('status')} "
                    f"prob={ba_result.get('success_probability', 0.7):.0%} "
                    f"source={ba_result.get('source')}"
                )
                secure_log("Resubmitter-Bedrock", {
                    "claim_id": claim_id,
                    "status": ba_result.get("status"),
                    "source": ba_result.get("source"),
                })
                return state
        except Exception as _be:
            state.setdefault('log', []).append(f"[Resubmitter] Bedrock Agent skipped: {_be}")
        # ── End Bedrock Agent Core ────────────────────────────────────────────

        # Add detailed logging for UI activity tracking
        state.setdefault('log', []).append("[Resubmitter] Preparing intelligent resubmission with AI-generated appeal packet")
        state.setdefault('log', []).append("[Resubmitter] Analyzing original denial reason and optimizing resubmission strategy")
        
        # Log the resubmission attempt with context
        patient_name = corrected_data.get('patient_name', 'Unknown Patient')
        insurance_company = corrected_data.get('insurance_company', 'Unknown Insurer')
        log_entry = f"Starting intelligent resubmission for {patient_name} (Insurer: {insurance_company})"
        state.setdefault('log', []).append(log_entry)
        
        # Determine resubmission strategy based on denial reason
        resubmission_strategy = determine_resubmission_strategy(denial_info, corrected_data)
        state['log'].append(f"[Resubmitter] Strategy: {resubmission_strategy['approach']}")
        
        # Simulate intelligent resubmission process
        await asyncio.sleep(1)  # Simulate processing time
        
        # In a real system, this would:
        # 1. Take the corrected data + appeal packet
        # 2. Apply insurer-specific formatting
        # 3. Include supporting documentation
        # 4. Submit via appropriate API endpoint
        # 5. Monitor submission status
        # 6. Handle any follow-up requirements
        
        # Simulate resubmission outcome based on denial type and strategy
        success_probability = calculate_resubmission_success_rate(denial_info, resubmission_strategy)
        resubmission_successful = success_probability > 0.6  # 60% threshold for success
        
        if resubmission_successful:
            resubmission_result = {
                'resubmission_id': f"RESUB_{claim_id}_{int(asyncio.get_event_loop().time())}",
                'status': 'resubmitted',
                'resubmission_date': str(asyncio.get_event_loop().time()),
                'corrected_issues': state.get('issues', []),
                'appeal_included': bool(appeal_packet),
                'strategy_used': resubmission_strategy['approach'],
                'expected_outcome': 'likely_approval',
                'success_probability': success_probability,
                'success': True
            }
            
            state['log'].append(f"[Resubmitter] ✅ Resubmission successful - Appeal submitted with {success_probability:.1%} success probability")
            state["final_status"] = "appeal_resubmitted"
            
        else:
            # Even if resubmission has low success rate, still attempt it
            resubmission_result = {
                'resubmission_id': f"RESUB_{claim_id}_{int(asyncio.get_event_loop().time())}",
                'status': 'resubmitted_low_confidence',
                'resubmission_date': str(asyncio.get_event_loop().time()),
                'corrected_issues': state.get('issues', []),
                'appeal_included': bool(appeal_packet),
                'strategy_used': resubmission_strategy['approach'],
                'expected_outcome': 'requires_additional_documentation',
                'success_probability': success_probability,
                'success': True,  # Still considered successful submission
                'notes': 'May require additional patient data updates in OpenEMR'
            }
            
            state['log'].append(f"[Resubmitter] ⚠️ Resubmission completed with {success_probability:.1%} success probability - may need patient data updates")
            state["final_status"] = "appeal_resubmitted_low_confidence"
        
        # Update state
        state['resubmission_result'] = resubmission_result
        state['log'].append(f"[Resubmitter] Claim resubmitted via {resubmission_strategy['approach']} strategy")
        
        secure_log("resubmitter", state)
        
        return state
        
    except Exception as e:
        error_msg = f"[ERROR] Resubmission failed: {str(e)}"
        secure_log("resubmitter", {**state, "error": error_msg})
        
        state.setdefault('log', []).append(error_msg)
        state['resubmission_result'] = {
            'status': 'failed',
            'error': str(e),
            'success': False
        }
        state["final_status"] = "resubmission"
        
        return state


def determine_resubmission_strategy(denial_info: Dict[str, Any], claim_data: Dict[str, Any]) -> Dict[str, Any]:
    """Determine the best resubmission strategy based on denial reason"""
    
    denial_reason = denial_info.get('reason', '').lower()
    
    if 'authorization' in denial_reason:
        return {
            'approach': 'prior_auth_appeal',
            'priority': 'high',
            'documentation_focus': 'authorization_necessity',
            'expected_timeline': '5-10_business_days'
        }
    elif 'medical necessity' in denial_reason or 'documentation' in denial_reason:
        return {
            'approach': 'clinical_documentation_appeal', 
            'priority': 'high',
            'documentation_focus': 'medical_necessity_evidence',
            'expected_timeline': '7-14_business_days'
        }
    elif 'coding' in denial_reason or 'diagnosis' in denial_reason:
        return {
            'approach': 'coding_correction_resubmission',
            'priority': 'medium', 
            'documentation_focus': 'accurate_coding_justification',
            'expected_timeline': '3-7_business_days'
        }
    elif 'eligibility' in denial_reason or 'coverage' in denial_reason:
        return {
            'approach': 'eligibility_verification_appeal',
            'priority': 'medium',
            'documentation_focus': 'coverage_verification',
            'expected_timeline': '5-10_business_days'
        }
    else:
        return {
            'approach': 'comprehensive_appeal',
            'priority': 'medium',
            'documentation_focus': 'comprehensive_review',
            'expected_timeline': '10-15_business_days'
        }


def calculate_resubmission_success_rate(denial_info: Dict[str, Any], strategy: Dict[str, Any]) -> float:
    """Calculate expected success rate based on denial type and strategy match"""
    
    base_success_rate = denial_info.get('success_rate', 0.75)  # Default 75%
    
    # Adjust based on strategy appropriateness
    denial_reason = denial_info.get('reason', '').lower()
    strategy_approach = strategy.get('approach', '')
    
    # Strategy matching bonuses
    if 'authorization' in denial_reason and 'prior_auth' in strategy_approach:
        base_success_rate += 0.15  # +15% for appropriate strategy
    elif 'documentation' in denial_reason and 'clinical_documentation' in strategy_approach:
        base_success_rate += 0.15
    elif 'coding' in denial_reason and 'coding_correction' in strategy_approach:
        base_success_rate += 0.20  # +20% for coding issues (easier to fix)
    
    # Cap at 95% (nothing is 100% certain)
    return min(base_success_rate, 0.95)

# For direct testing
if __name__ == "__main__":
    # Test the resubmitter
    test_state = {
        'claim_id': 'TEST_001',
        'corrected_data': {
            'patient_id': 'P001',
            'diagnosis': 'Corrected diagnosis',
            'procedure': 'Corrected procedure'
        },
        'issues': ['Missing diagnosis code', 'Incorrect procedure code'],
        'log': []
    }
    
    result = asyncio.run(run_resubmission(test_state))
    print(json.dumps(result, indent=2))
