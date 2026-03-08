"""
bedrock_agent_integration.py
============================
Provides clean integration between LangGraph agents and Bedrock Agent Core.

Each function calls the appropriate Bedrock Agent, parses the response,
and returns structured data that the LangGraph agent can use directly.

If the Bedrock Agent call fails or returns empty, returns None so the
calling agent can fall back to its existing local logic.
"""

import json
import logging
import re
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def _parse_json_from_text(text: str) -> Optional[dict]:
    """Extract first JSON object from agent response text."""
    if not text:
        return None
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try extracting JSON block
    match = re.search(r'\{[\s\S]+\}', text)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


# ── Risk Predictor ────────────────────────────────────────────────────────────

def bedrock_risk_assessment(claim: dict) -> Optional[Dict[str, Any]]:
    """
    Call RiskPredictor Bedrock Agent.
    Returns dict with risk_score, issues, recommendations or None on failure.
    """
    try:
        from tools.bedrock_agent_client import invoke_risk_predictor
        response = invoke_risk_predictor(claim, session_id=claim.get("claim_id"))
        if not response:
            return None

        parsed = _parse_json_from_text(response)
        if parsed:
            return {
                "risk_score":      float(parsed.get("risk_score", 0.5)),
                "issues":          parsed.get("issues", []),
                "recommendations": parsed.get("recommendations", []),
                "confidence":      float(parsed.get("confidence", 0.7)),
                "source":          "bedrock_agent",
                "raw_response":    response[:500],
            }

        # If no JSON, extract risk score from text
        score_match = re.search(r'risk[_\s]score[:\s]+([0-9.]+)', response, re.IGNORECASE)
        risk_score = float(score_match.group(1)) if score_match else 0.5

        return {
            "risk_score":      risk_score,
            "issues":          [],
            "recommendations": [],
            "confidence":      0.6,
            "source":          "bedrock_agent_text",
            "raw_response":    response[:500],
        }
    except Exception as e:
        logger.warning(f"[BedrockIntegration] risk_assessment failed: {e}")
        return None


# ── Auto Corrector ────────────────────────────────────────────────────────────

def bedrock_auto_correct(claim: dict, issues: list) -> Optional[Dict[str, Any]]:
    """
    Call AutoCorrector Bedrock Agent.
    Returns dict with corrected_data fields or None on failure.
    """
    try:
        from tools.bedrock_agent_client import invoke_auto_corrector
        response = invoke_auto_corrector(claim, issues, session_id=claim.get("claim_id"))
        if not response:
            return None

        parsed = _parse_json_from_text(response)
        if parsed:
            return {
                "corrected_icd":   parsed.get("corrected_code") or parsed.get("icd_code"),
                "corrected_cpt":   parsed.get("corrected_cpt") or parsed.get("cpt_code"),
                "prior_auth":      parsed.get("prior_auth_number") or parsed.get("prior_auth"),
                "npi_valid":       parsed.get("valid", True),
                "corrections":     parsed.get("corrections", []),
                "source":          "bedrock_agent",
                "raw_response":    response[:500],
            }

        return {
            "source":       "bedrock_agent_text",
            "raw_response": response[:500],
        }
    except Exception as e:
        logger.warning(f"[BedrockIntegration] auto_correct failed: {e}")
        return None


# ── Claim Submitter ───────────────────────────────────────────────────────────

def bedrock_submit_claim(claim: dict) -> Optional[Dict[str, Any]]:
    """
    Call ClaimSubmitter Bedrock Agent.
    Returns dict with status, claim_id, denial_info or None on failure.
    """
    try:
        from tools.bedrock_agent_client import invoke_claim_submitter
        response = invoke_claim_submitter(claim, session_id=claim.get("claim_id"))
        if not response:
            return None

        parsed = _parse_json_from_text(response)
        if parsed:
            return {
                "status":      parsed.get("status", "submitted"),
                "claim_id":    parsed.get("claim_id", claim.get("claim_id")),
                "denial_info": parsed.get("denial_info"),
                "eligible":    parsed.get("eligible", True),
                "source":      "bedrock_agent",
                "raw_response": response[:500],
            }

        # Parse status from text
        status = "approved" if "approved" in response.lower() else \
                 "denied"   if "denied"   in response.lower() else "submitted"
        return {
            "status":       status,
            "claim_id":     claim.get("claim_id"),
            "source":       "bedrock_agent_text",
            "raw_response": response[:500],
        }
    except Exception as e:
        logger.warning(f"[BedrockIntegration] submit_claim failed: {e}")
        return None


# ── Appeal Generator ──────────────────────────────────────────────────────────

def bedrock_generate_appeal(claim: dict, denial_reason: str) -> Optional[Dict[str, Any]]:
    """
    Call AppealGenerator Bedrock Agent.
    Returns dict with appeal_text, s3_key or None on failure.
    """
    try:
        from tools.bedrock_agent_client import invoke_appeal_generator
        response = invoke_appeal_generator(claim, denial_reason,
                                           session_id=claim.get("claim_id"))
        if not response:
            return None

        parsed = _parse_json_from_text(response)
        if parsed:
            return {
                "appeal_text": parsed.get("appeal_text", response),
                "s3_key":      parsed.get("appeal_s3_key") or parsed.get("s3_key"),
                "word_count":  parsed.get("word_count", 0),
                "source":      "bedrock_agent",
            }

        # Raw text is the appeal letter
        return {
            "appeal_text": response.strip(),
            "s3_key":      None,
            "word_count":  len(response.split()),
            "source":      "bedrock_agent_text",
        }
    except Exception as e:
        logger.warning(f"[BedrockIntegration] generate_appeal failed: {e}")
        return None


# ── Resubmitter ───────────────────────────────────────────────────────────────

def bedrock_resubmit(claim: dict, appeal_text: str) -> Optional[Dict[str, Any]]:
    """
    Call Resubmitter Bedrock Agent.
    Returns dict with status, success_probability or None on failure.
    """
    try:
        from tools.bedrock_agent_client import invoke_resubmitter
        response = invoke_resubmitter(claim, appeal_text,
                                       session_id=claim.get("claim_id"))
        if not response:
            return None

        parsed = _parse_json_from_text(response)
        if parsed:
            return {
                "status":               parsed.get("status", "resubmitted"),
                "resubmission_id":      parsed.get("resubmission_id"),
                "success_probability":  float(parsed.get("success_probability", 0.7)),
                "strategy":             parsed.get("strategy_type", "standard_appeal"),
                "source":               "bedrock_agent",
            }

        status = "approved" if "approved" in response.lower() else "resubmitted"
        return {
            "status":              status,
            "success_probability": 0.7,
            "source":              "bedrock_agent_text",
        }
    except Exception as e:
        logger.warning(f"[BedrockIntegration] resubmit failed: {e}")
        return None


# ── Feedback Learner ──────────────────────────────────────────────────────────

def bedrock_learn_outcome(claim: dict, outcome: str) -> Optional[Dict[str, Any]]:
    """
    Call FeedbackLearner Bedrock Agent.
    Returns dict with patterns_updated, insights or None on failure.
    """
    try:
        from tools.bedrock_agent_client import invoke_feedback_learner
        response = invoke_feedback_learner(claim, outcome,
                                            session_id=claim.get("claim_id"))
        if not response:
            return None

        parsed = _parse_json_from_text(response)
        if parsed:
            return {
                "patterns_updated": parsed.get("updated", parsed.get("patterns_updated", True)),
                "insights":         parsed.get("recommendation") or parsed.get("insights", ""),
                "success_rate":     parsed.get("historical_success_rate"),
                "source":           "bedrock_agent",
            }

        return {
            "patterns_updated": True,
            "insights":         response[:300],
            "source":           "bedrock_agent_text",
        }
    except Exception as e:
        logger.warning(f"[BedrockIntegration] learn_outcome failed: {e}")
        return None
