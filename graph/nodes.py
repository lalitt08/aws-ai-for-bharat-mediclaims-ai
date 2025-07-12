# graph/nodes.py
import sys
import os
# Ensure project root is in sys.path for direct execution
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import sys
import os
try:
    from agents.risk_predictor import run_risk_prediction
    from agents.auto_corrector import run_auto_correction
    from agents.claim_submitter import run_claim_submission
    from agents.appeal_generator import run_appeal_generation
    from agents.resubmitter import run_resubmission
    from agents.feedback_learner import run_feedback_learning
except ModuleNotFoundError:
    # If running from graph/ directory, add parent to sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agents.risk_predictor import run_risk_prediction
    from agents.auto_corrector import run_auto_correction
    from agents.claim_submitter import run_claim_submission
    from agents.appeal_generator import run_appeal_generation
    from agents.resubmitter import run_resubmission
    from agents.feedback_learner import run_feedback_learning


# Wrap each agent as a LangGraph-compatible async node
async def risk_predictor_node(state: dict) -> dict:
    return await run_risk_prediction(state)

async def auto_corrector_node(state: dict) -> dict:
    return await run_auto_correction(state)

async def claim_submitter_node(state: dict) -> dict:
    return await run_claim_submission(state)

async def appeal_generator_node(state: dict) -> dict:
    return await run_appeal_generation(state)

async def resubmitter_node(state: dict) -> dict:
    return await run_resubmission(state)

async def feedback_learner_node(state: dict) -> dict:
    return await run_feedback_learning(state)
