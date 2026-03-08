"""
ERA (Electronic Remittance Advice) Processor Service
Parses real ERA/835 files and uses Azure OpenAI LLM for intelligent analysis.
Supports standard X12 835 format with CLP, CAS, NM1, SVC, DTM, N1 segments.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# ── Azure OpenAI setup ──
_llm_client = None

def _get_llm_client():
    """Lazy-load Bedrock client using bearer token from .env."""
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        load_dotenv(env_path)

        import requests as _req
        token = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")
        region = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
        model = os.getenv("AWS_BEDROCK_MODEL_ID", "meta.llama3-70b-instruct-v1:0")

        if not token:
            raise ValueError("AWS_BEARER_TOKEN_BEDROCK not set")

        # Store as a simple callable dict
        _llm_client = {"token": token, "region": region, "model": model, "requests": _req}
        logger.info(f"Bedrock client initialized: {model} in {region}")
    except Exception as e:
        logger.warning(f"Could not init Bedrock client: {e}. LLM analysis will be skipped.")
        _llm_client = None
    return _llm_client


def _get_deployment_name() -> str:
    return os.getenv("AWS_BEDROCK_MODEL_ID", "meta.llama3-70b-instruct-v1:0")


class ERAProcessor:
    """Service for processing ERA/835 files with LLM-powered analysis."""

    def __init__(self):
        self.supported_formats = ["835", "ERA", "XML", "JSON", "TXT", "EDI"]
        # CARC (Claim Adjustment Reason Code) knowledge base
        self.denial_codes = {
            "1": "Deductible amount",
            "2": "Coinsurance amount",
            "3": "Co-payment amount",
            "4": "The procedure code is inconsistent with the modifier used",
            "5": "The procedure code/bill type is inconsistent with the place of service",
            "16": "Claim/service lacks information needed for adjudication",
            "18": "Exact duplicate claim/service",
            "22": "This care may be covered by another payer per coordination of benefits",
            "23": "The impact of prior payer(s) adjudication including payments and/or adjustments",
            "27": "Expenses incurred after coverage terminated",
            "29": "The time limit for filing has expired",
            "45": "Charge exceeds fee schedule/maximum allowable",
            "50": "These are non-covered services because this is not deemed a medical necessity",
            "96": "Non-covered charge(s)",
            "97": "The benefit for this service is included in the payment for another service",
            "109": "Claim/service not covered by this payer/contractor",
            "119": "Benefit maximum for this time period or occurrence has been reached",
            "151": "Information submitted does not support this many/frequency of services",
            "197": "Precertification/authorization/notification absent",
            "204": "This service/equipment/drug is not covered under the patient benefit plan",
            "242": "Services not provided by network/primary care providers",
            "B7": "Provider not certified/eligible to be paid for this procedure/service",
        }

    def process_era_file(self, file_content: str, filename: str) -> Dict[str, Any]:
        """Parse a real ERA/835 file, extract denials, then run LLM analysis."""

        # Step 1 — deterministic parsing of X12 segments
        segments = self._split_segments(file_content)
        claims = self._parse_segments(segments)

        denials_extracted: List[Dict[str, Any]] = []
        paid_count = denied_count = pending_count = 0

        for claim in claims:
            adjustments = claim.get("adjustments", [])
            status_code = claim.get("claim_status", "")
            charged = claim.get("charged_amount", 0)
            paid = claim.get("paid_amount", 0)

            if adjustments:
                for adj in adjustments:
                    code_num = adj.get("reason_code", "16")
                    code_key = f"CO-{code_num}"
                    description = self.denial_codes.get(
                        str(code_num), f"Adjustment reason code {code_num}"
                    )
                    denials_extracted.append({
                        "claim_id": claim.get("claim_id", ""),
                        "patient_id": claim.get("patient_id", ""),
                        "patient_name": claim.get("patient_name", ""),
                        "service_date": claim.get("service_date", ""),
                        "denial_code": code_key,
                        "denial_reason": description,
                        "denied_amount": adj.get("amount", 0),
                        "payer": claim.get("payer", "Unknown"),
                        "provider_id": claim.get("provider_id", ""),
                        "procedure_code": claim.get("procedure_code", ""),
                        "diagnosis_code": claim.get("diagnosis_code", ""),
                        "charged_amount": charged,
                        "paid_amount": paid,
                    })
                denied_count += 1
            elif paid > 0 or status_code in ("1", "19"):
                paid_count += 1
            elif status_code in ("2", "4", "22"):
                denied_count += 1
            else:
                pending_count += 1

        total_claims = max(len(claims), denied_count + paid_count + pending_count, 1)

        result = {
            "file_id": f"ERA-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "filename": filename,
            "processed_at": datetime.now().isoformat(),
            "status": "completed",
            "summary": {
                "total_claims": total_claims,
                "paid_claims": paid_count,
                "denied_claims": denied_count,
                "pending_claims": pending_count,
            },
            "denials_extracted": denials_extracted,
            "errors": [],
        }

        # Step 2 — LLM-powered intelligent analysis
        llm_analysis = self._run_llm_analysis(denials_extracted, result["summary"])
        if llm_analysis:
            result["llm_analysis"] = llm_analysis

        # Step 3 — Upload ERA file to S3 for persistent storage
        try:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            from tools.s3_storage import upload_era
            patient_id = denials_extracted[0].get("patient_id", "UNKNOWN") if denials_extracted else "UNKNOWN"
            upload_era(patient_id, filename, file_content.encode("utf-8"))
        except Exception as s3_err:
            pass  # S3 upload is best-effort

        return result

    # ── LLM Analysis ──────────────────────────────────────────────────

    def _run_llm_analysis(self, denials: List[Dict], summary: Dict) -> Optional[Dict]:
        """Call AWS Bedrock (bearer token) to produce an intelligent narrative analysis of the ERA."""
        client = _get_llm_client()
        if not client or not denials:
            return None

        # Build a concise denial summary for the prompt
        denial_lines = []
        for d in denials[:10]:  # cap to avoid token overflow
            denial_lines.append(
                f"- Claim {d['claim_id']}: {d['denial_code']} — {d['denial_reason']} "
                f"(${d['denied_amount']:.2f}, procedure {d['procedure_code']}, "
                f"patient {d['patient_name'] or d['patient_id']})"
            )
        denial_text = "\n".join(denial_lines)

        prompt = f"""You are an expert medical billing analyst AI. Analyze the following ERA/835 denial data and provide actionable insights.

ERA Summary:
- Total claims: {summary.get('total_claims', 0)}
- Paid: {summary.get('paid_claims', 0)}
- Denied: {summary.get('denied_claims', 0)}

Denied Claims:
{denial_text}

Respond in valid JSON with these keys:
{{
  "overall_assessment": "2-3 sentence summary of the denial pattern",
  "root_cause_analysis": "What is the primary root cause across these denials",
  "risk_level": "high/medium/low",
  "top_recommendations": ["list of 3 specific actionable steps"],
  "estimated_recovery": "dollar estimate of recoverable amount with appeals",
  "appeal_priority_order": ["ordered list of claim IDs to appeal first"],
  "process_improvement": "one suggestion to prevent future denials like these"
}}
Only output the JSON, no markdown fences."""

        try:
            req = client["requests"]
            token = client["token"]
            region = client["region"]
            model = client["model"]
            url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model}/invoke"

            # Build Llama 3 instruct prompt
            full_prompt = (
                "<|begin_of_text|>"
                "<|start_header_id|>system<|end_header_id|>\n"
                "You are a healthcare revenue cycle management AI expert."
                "<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n"
                f"{prompt}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n"
            )
            payload = {"prompt": full_prompt, "max_gen_len": 800, "temperature": 0.3}

            resp = req.post(
                url,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.json().get("generation", "").strip()

            # Strip markdown fences if model adds them
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            return {"error": str(e), "fallback": True}

    # ── X12 Segment Parsing ───────────────────────────────────────────

    def _split_segments(self, content: str) -> List[str]:
        """Split ERA content into individual segments."""
        content = content.lstrip("\ufeff")
        if "~" in content:
            raw = content.replace("\r\n", "").replace("\n", "").replace("\r", "")
            return [s.strip() for s in raw.split("~") if s.strip()]
        return [s.strip() for s in content.split("\n") if s.strip()]

    def _parse_segments(self, segments: List[str]) -> List[Dict[str, Any]]:
        """Walk through segments and build structured claim records."""
        claims: List[Dict[str, Any]] = []
        current_payer = ""
        current_claim: Optional[Dict[str, Any]] = None

        for seg in segments:
            elements = seg.split("*")
            seg_id = elements[0].strip() if elements else ""

            if seg_id == "N1" and len(elements) > 2 and elements[1] == "PR":
                current_payer = elements[2]

            elif seg_id == "CLP" and len(elements) > 3:
                if current_claim:
                    claims.append(current_claim)
                current_claim = {
                    "claim_id": elements[1] if len(elements) > 1 else "",
                    "claim_status": elements[2] if len(elements) > 2 else "",
                    "charged_amount": self._safe_float(elements[3]) if len(elements) > 3 else 0,
                    "paid_amount": self._safe_float(elements[4]) if len(elements) > 4 else 0,
                    "payer": current_payer,
                    "patient_id": elements[7] if len(elements) > 7 else "",
                    "patient_name": "",
                    "service_date": "",
                    "procedure_code": "",
                    "diagnosis_code": "",
                    "provider_id": "",
                    "adjustments": [],
                }

            elif seg_id == "CAS" and current_claim and len(elements) > 2:
                group_code = elements[1]
                i = 2
                while i < len(elements):
                    reason_code = elements[i] if i < len(elements) else ""
                    amount = self._safe_float(elements[i + 1]) if (i + 1) < len(elements) else 0
                    if reason_code:
                        current_claim["adjustments"].append({
                            "group_code": group_code,
                            "reason_code": reason_code,
                            "amount": amount,
                        })
                    i += 3

            elif seg_id == "NM1" and current_claim and len(elements) > 3:
                qualifier = elements[1]
                if qualifier == "QC":
                    last = elements[3] if len(elements) > 3 else ""
                    first = elements[4] if len(elements) > 4 else ""
                    current_claim["patient_name"] = f"{first} {last}".strip()
                    if len(elements) > 9:
                        current_claim["patient_id"] = current_claim["patient_id"] or elements[9]
                elif qualifier == "82" and len(elements) > 9:
                    current_claim["provider_id"] = elements[9]

            elif seg_id == "SVC" and current_claim and len(elements) > 1:
                svc_id = elements[1]
                current_claim["procedure_code"] = svc_id.split(":")[1] if ":" in svc_id else svc_id

            elif seg_id == "DTM" and current_claim and len(elements) > 2:
                if elements[1] in ("232", "233", "472"):
                    current_claim["service_date"] = self._format_date(elements[2])

        if current_claim:
            claims.append(current_claim)
        return claims

    # ── Helpers ────────────────────────────────────────────────────────

    def _safe_float(self, value: str) -> float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _format_date(self, raw: str) -> str:
        raw = raw.strip()
        if len(raw) == 8 and raw.isdigit():
            return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
        return raw

    # ── Classification / analytics helpers ─────────────────────────────

    def parse_denial_codes(self, era_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        categorized = []
        for denial in era_data.get("denials_extracted", []):
            code = denial.get("denial_code", "")
            categorized.append({
                **denial,
                "category": self._categorize_denial_code(code),
                "severity": self._assess_denial_severity(code),
                "appeal_likelihood": self._calculate_appeal_likelihood(code, denial),
                "suggested_actions": self._get_suggested_actions(code),
                "required_documentation": self._get_required_documentation(code),
            })
        return categorized

    def _categorize_denial_code(self, denial_code: str) -> str:
        m = {
            "CO-4": "coding_error", "CO-5": "coding_error",
            "CO-16": "documentation", "CO-18": "duplicate_claim",
            "CO-27": "eligibility", "CO-29": "timely_filing",
            "CO-45": "fee_schedule", "CO-50": "medical_necessity",
            "CO-96": "medical_necessity", "CO-97": "bundling",
            "CO-109": "eligibility", "CO-119": "benefit_maximum",
            "CO-151": "medical_necessity", "CO-197": "prior_authorization",
            "CO-204": "policy_exclusion", "CO-242": "network",
        }
        return m.get(denial_code, "other")

    def _assess_denial_severity(self, denial_code: str) -> str:
        m = {
            "CO-4": "medium", "CO-5": "medium", "CO-16": "medium",
            "CO-18": "low", "CO-27": "high", "CO-29": "high",
            "CO-45": "low", "CO-50": "medium", "CO-96": "medium",
            "CO-97": "low", "CO-109": "high", "CO-119": "high",
            "CO-151": "medium", "CO-197": "low", "CO-204": "high",
            "CO-242": "medium",
        }
        return m.get(denial_code, "medium")

    def _calculate_appeal_likelihood(self, denial_code: str, denial_data: Dict[str, Any]) -> float:
        base = {
            "CO-4": 0.85, "CO-5": 0.80, "CO-16": 0.75, "CO-18": 0.70,
            "CO-27": 0.25, "CO-29": 0.15, "CO-45": 0.60, "CO-50": 0.60,
            "CO-96": 0.60, "CO-97": 0.85, "CO-109": 0.20, "CO-119": 0.30,
            "CO-151": 0.65, "CO-197": 0.80, "CO-204": 0.30, "CO-242": 0.50,
        }
        likelihood = base.get(denial_code, 0.50)
        amount = denial_data.get("denied_amount", 0)
        if amount > 2000:
            likelihood += 0.10
        elif amount < 500:
            likelihood -= 0.05
        return round(max(0.0, min(1.0, likelihood)), 2)

    def _get_suggested_actions(self, denial_code: str) -> List[str]:
        a = {
            "CO-4": ["Review and correct procedure/modifier codes", "Verify code-to-diagnosis linkage", "Resubmit corrected claim"],
            "CO-16": ["Submit missing documentation", "Verify all required fields", "Contact provider for additional info"],
            "CO-18": ["Verify not a true duplicate", "Add distinguishing modifier", "Document separate encounter"],
            "CO-27": ["Verify patient eligibility dates", "Check if emergency/urgent", "Appeal with eligibility docs"],
            "CO-29": ["Document good cause for late filing", "Check timely filing rules", "Submit appeal with explanation"],
            "CO-50": ["Provide medical necessity documentation", "Submit clinical guidelines", "Request peer-to-peer review"],
            "CO-97": ["Verify bundling rules", "Add modifier if unbundling justified", "Submit supporting docs"],
            "CO-197": ["Obtain retroactive prior auth", "Document emergency nature", "Submit auth request with clinical justification"],
            "CO-204": ["Review benefit plan details", "Check alternative covered services", "Appeal with medical necessity"],
        }
        return a.get(denial_code, ["Review denial reason", "Gather supporting documentation", "Submit formal appeal"])

    def _get_required_documentation(self, denial_code: str) -> List[str]:
        d = {
            "CO-4": ["Correct CPT/HCPCS codes", "Operative notes", "Modifier justification"],
            "CO-16": ["Complete medical records", "Provider notes", "Test results and imaging"],
            "CO-27": ["Eligibility verification", "Coverage dates", "Emergency treatment docs"],
            "CO-29": ["Original submission records", "Filing delay documentation", "Good cause letter"],
            "CO-50": ["Clinical notes for medical necessity", "Treatment plans", "Published guidelines"],
            "CO-197": ["Prior auth request form", "Clinical justification letter", "Supporting medical records"],
        }
        return d.get(denial_code, ["Original claim docs", "Medical records", "Appeal justification letter"])

    def get_era_statistics(self, era_data: Dict[str, Any]) -> Dict[str, Any]:
        denials = era_data.get("denials_extracted", [])
        if not denials:
            return {"total_denials": 0, "total_denied_amount": 0}
        total_denied = sum(d.get("denied_amount", 0) for d in denials)
        cats: Dict[str, int] = {}
        codes: Dict[str, int] = {}
        payers: Dict[str, int] = {}
        for d in denials:
            c = self._categorize_denial_code(d.get("denial_code", ""))
            cats[c] = cats.get(c, 0) + 1
            codes[d.get("denial_code", "")] = codes.get(d.get("denial_code", ""), 0) + 1
            p = d.get("payer", "Unknown")
            payers[p] = payers.get(p, 0) + 1
        return {
            "total_denials": len(denials),
            "total_denied_amount": round(total_denied, 2),
            "average_denied_amount": round(total_denied / len(denials), 2),
            "denial_categories": cats,
            "denial_codes": codes,
            "payers": payers,
            "statistics_generated_at": datetime.now().isoformat(),
        }
