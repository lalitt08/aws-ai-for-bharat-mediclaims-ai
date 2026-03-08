"""
Bedrock Agent Client
====================
Invokes AWS Bedrock Agents (Agent Core) for each step of the claims pipeline.
Each of the 6 LangGraph agents has a corresponding Bedrock Agent with Lambda Action Groups.

Agent IDs (us-east-1, account 390783052961):
  RiskPredictor    XLAYW801JO
  AppealGenerator  S4YKZVC69F
  AutoCorrector    53KNQP1PMD
  ClaimSubmitter   BSI3CF17OU
  Resubmitter      VCKGXKAZN0
  FeedbackLearner  WUKCB3RG8Z

After running aws_setup/deploy_lambdas.py, each agent gets a 'live' alias.
Set BEDROCK_AGENT_ALIAS_<KEY> in .env with the alias IDs printed by that script.
Falls back to TSTALIASID (test alias) if per-agent alias not set.
"""

import boto3
import json
import uuid
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

REGION       = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
DEFAULT_ALIAS = os.getenv("BEDROCK_AGENT_ALIAS", "TSTALIASID")

# Agent registry — IDs
AGENT_IDS: Dict[str, str] = {
    "risk_predictor":   os.getenv("BEDROCK_AGENT_RISK_PREDICTOR",   "XLAYW801JO"),
    "appeal_generator": os.getenv("BEDROCK_AGENT_APPEAL_GENERATOR", "S4YKZVC69F"),
    "auto_corrector":   os.getenv("BEDROCK_AGENT_AUTO_CORRECTOR",   "53KNQP1PMD"),
    "claim_submitter":  os.getenv("BEDROCK_AGENT_CLAIM_SUBMITTER",  "BSI3CF17OU"),
    "resubmitter":      os.getenv("BEDROCK_AGENT_RESUBMITTER",      "VCKGXKAZN0"),
    "feedback_learner": os.getenv("BEDROCK_AGENT_FEEDBACK_LEARNER", "WUKCB3RG8Z"),
}

# Live aliases — v4, Nova Micro + Action Groups, IAM key auth
AGENT_ALIASES: Dict[str, str] = {
    "risk_predictor":   os.getenv("BEDROCK_AGENT_ALIAS_RISK_PREDICTOR",   "GMML5M64KW"),
    "appeal_generator": os.getenv("BEDROCK_AGENT_ALIAS_APPEAL_GENERATOR", "F7V98MQXIZ"),
    "auto_corrector":   os.getenv("BEDROCK_AGENT_ALIAS_AUTO_CORRECTOR",   "FYFALB8NSI"),
    "claim_submitter":  os.getenv("BEDROCK_AGENT_ALIAS_CLAIM_SUBMITTER",  "BZBIFWBIBO"),
    "resubmitter":      os.getenv("BEDROCK_AGENT_ALIAS_RESUBMITTER",      "EWC6FLUFBK"),
    "feedback_learner": os.getenv("BEDROCK_AGENT_ALIAS_FEEDBACK_LEARNER", "U9I9LCMJIL"),
}


def _runtime_client():
    return boto3.client("bedrock-agent-runtime", region_name=REGION)


def invoke_agent(agent_key: str, prompt: str, session_id: str = None,
                 timeout: int = 30) -> str:
    """
    Invoke a Bedrock Agent and return the full text response.

    With Lambda Action Groups attached, the agent will:
    1. Receive the prompt
    2. Decide which tool(s) to call
    3. Invoke the Lambda function(s)
    4. Synthesize a final response

    Returns the final synthesized text, or empty string on error.
    """
    agent_id  = AGENT_IDS.get(agent_key)
    alias_id  = AGENT_ALIASES.get(agent_key, DEFAULT_ALIAS)

    if not agent_id:
        logger.warning(f"[BedrockAgent] Unknown agent key: {agent_key}")
        return ""

    session_id = session_id or str(uuid.uuid4())

    try:
        client = _runtime_client()
        response = client.invoke_agent(
            agentId=agent_id,
            agentAliasId=alias_id,
            sessionId=session_id,
            inputText=prompt,
        )

        # Consume the streaming EventStream
        full_text = ""
        completion = response.get("completion")
        if completion is None:
            logger.warning(f"[BedrockAgent] {agent_key}: no 'completion' in response")
            return ""

        for event in completion:
            # Text chunk from agent
            chunk = event.get("chunk", {})
            if "bytes" in chunk:
                full_text += chunk["bytes"].decode("utf-8")

            # Trace events (optional debug)
            trace = event.get("trace", {})
            if trace:
                trace_type = list(trace.keys())[0] if trace else ""
                logger.debug(f"[BedrockAgent] {agent_key} trace: {trace_type}")

            # Return-of-control (agent wants caller to handle tool)
            roc = event.get("returnControl", {})
            if roc:
                logger.info(f"[BedrockAgent] {agent_key} returnControl: {json.dumps(roc)[:200]}")

        logger.info(f"[BedrockAgent] {agent_key} → {len(full_text)} chars")
        return full_text.strip()

    except Exception as e:
        logger.warning(f"[BedrockAgent] {agent_key} invocation failed: {e}")
        return ""


# ── Typed wrappers for each agent ─────────────────────────────────────────────

def invoke_risk_predictor(claim: dict, session_id: str = None) -> str:
    prompt = (
        f"Analyze this medical claim for denial risk.\n"
        f"Patient ID: {claim.get('patient_id')}\n"
        f"Insurer: {claim.get('insurer') or claim.get('insurance_company')}\n"
        f"CPT Code: {claim.get('cpt_code') or claim.get('procedure_code')}\n"
        f"ICD-10: {claim.get('icd_code') or claim.get('diagnosis_code')}\n"
        f"Amount: ${claim.get('claim_amount', 0)}\n"
        f"Prior Auth: {claim.get('prior_auth', 'None')}\n\n"
        f"Use your tools to: get patient data, validate ICD-10 and CPT codes, "
        f"check prior authorization, and analyze denial patterns. "
        f"Return a risk score (0.0-1.0), list of issues, and recommendations."
    )
    return invoke_agent("risk_predictor", prompt, session_id)


def invoke_auto_corrector(claim: dict, issues: list, session_id: str = None) -> str:
    issues_text = "\n".join(f"- {i}" for i in issues) if issues else "- General review needed"
    prompt = (
        f"Auto-correct this medical claim. Issues identified:\n{issues_text}\n\n"
        f"Patient ID: {claim.get('patient_id')}\n"
        f"ICD-10: {claim.get('icd_code') or claim.get('diagnosis_code')}\n"
        f"CPT: {claim.get('cpt_code') or claim.get('procedure_code')}\n"
        f"Insurer: {claim.get('insurer') or claim.get('insurance_company')}\n"
        f"Provider NPI: {claim.get('provider_npi', '')}\n\n"
        f"Use your tools to correct ICD-10/CPT codes, generate prior auth if needed, "
        f"and validate the provider NPI. Return the corrected claim data."
    )
    return invoke_agent("auto_corrector", prompt, session_id)


def invoke_claim_submitter(claim: dict, session_id: str = None) -> str:
    insurer = claim.get("insurer") or claim.get("insurance_company", "")
    prompt = (
        f"Submit this medical claim to the insurer.\n"
        f"Patient ID: {claim.get('patient_id')}\n"
        f"Insurer: {insurer}\n"
        f"Claim Amount: ${claim.get('claim_amount', 0)}\n"
        f"Service Date: {claim.get('service_date', 'today')}\n\n"
        f"Use your tools to: check patient eligibility, submit the claim to the insurer, "
        f"and save the result. Return the submission status and any denial information."
    )
    return invoke_agent("claim_submitter", prompt, session_id)


def invoke_appeal_generator(claim: dict, denial_reason: str,
                              session_id: str = None) -> str:
    prompt = (
        f"Generate a formal appeal for this denied claim.\n"
        f"Patient ID: {claim.get('patient_id')}\n"
        f"Claim ID: {claim.get('claim_id', 'N/A')}\n"
        f"Insurer: {claim.get('insurer') or claim.get('insurance_company')}\n"
        f"Denial Reason: {denial_reason}\n"
        f"CPT: {claim.get('cpt_code') or claim.get('procedure_code')}\n"
        f"ICD-10: {claim.get('icd_code') or claim.get('diagnosis_code')}\n\n"
        f"Use your tools to: get denial details, check appeal requirements, "
        f"generate the appeal letter, and save it to S3. "
        f"Return the complete appeal letter text."
    )
    return invoke_agent("appeal_generator", prompt, session_id)


def invoke_resubmitter(claim: dict, appeal_text: str,
                        session_id: str = None) -> str:
    denial_code = claim.get("denial_code", "CO-16")
    denial_reason = claim.get("denial_reason", "")
    prompt = (
        f"Resubmit this denied claim with the appeal.\n"
        f"Patient ID: {claim.get('patient_id')}\n"
        f"Claim ID: {claim.get('claim_id', 'N/A')}\n"
        f"Insurer: {claim.get('insurer') or claim.get('insurance_company')}\n"
        f"Denial Code: {denial_code}\n"
        f"Denial Reason: {denial_reason}\n\n"
        f"Use your tools to: determine the resubmission strategy, "
        f"resubmit with the appeal, and update the claim status. "
        f"Return the resubmission result and final status."
    )
    return invoke_agent("resubmitter", prompt, session_id)


def invoke_feedback_learner(claim: dict, outcome: str,
                              session_id: str = None) -> str:
    prompt = (
        f"Learn from this claim outcome.\n"
        f"Patient ID: {claim.get('patient_id')}\n"
        f"Claim ID: {claim.get('claim_id', 'N/A')}\n"
        f"Outcome: {outcome}\n"
        f"Insurer: {claim.get('insurer') or claim.get('insurance_company')}\n"
        f"CPT: {claim.get('cpt_code') or claim.get('procedure_code')}\n"
        f"ICD-10: {claim.get('icd_code') or claim.get('diagnosis_code')}\n\n"
        f"Use your tools to: record the claim outcome, update denial patterns, "
        f"and retrieve learning insights. Return patterns identified and recommendations."
    )
    return invoke_agent("feedback_learner", prompt, session_id)


def check_agents_have_action_groups() -> Dict[str, bool]:
    """
    Utility: check which agents have Action Groups attached.
    Returns dict of agent_key → has_action_groups (bool).
    """
    bc = boto3.client("bedrock-agent", region_name=REGION)
    results = {}
    for key, agent_id in AGENT_IDS.items():
        try:
            ags = bc.list_agent_action_groups(
                agentId=agent_id, agentVersion="DRAFT"
            ).get("actionGroupSummaries", [])
            results[key] = len(ags) > 0
        except Exception as e:
            results[key] = False
            logger.warning(f"[BedrockAgent] Could not check {key}: {e}")
    return results
