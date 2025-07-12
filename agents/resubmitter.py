# agents/resubmitter.py - Resubmission Agent

import asyncio
import json
from typing import Dict, Any, List
from tools.logger import secure_log

async def run_resubmission(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resubmission Agent - Handles claim resubmission after corrections
    """
    secure_log("resubmitter", state)
    
    try:
        claim_id = state.get('claim_id', 'unknown')
        corrected_data = state.get('corrected_data', {})
        
        # Log the resubmission attempt
        log_entry = f"Resubmission attempt for claim {claim_id}"
        state.setdefault('log', []).append(log_entry)
        
        # Simulate resubmission process
        await asyncio.sleep(1)  # Simulate processing time
        
        # In a real system, this would:
        # 1. Take the corrected data
        # 2. Reformat it according to insurer requirements
        # 3. Submit to the insurer's system
        # 4. Handle any additional corrections needed
        
        # For now, simulate a successful resubmission
        resubmission_result = {
            'resubmission_id': f"RESUB_{claim_id}_{int(asyncio.get_event_loop().time())}",
            'status': 'resubmitted',
            'resubmission_date': str(asyncio.get_event_loop().time()),
            'corrected_issues': state.get('issues', []),
            'success': True
        }
        
        # Update state
        state['resubmission_result'] = resubmission_result
        state['log'].append(f"[SUCCESS] Resubmission completed: {resubmission_result['resubmission_id']}")
        
        # Set processing status
        state["final_status"] = "resubmitted"
        
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
        
        return state

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
