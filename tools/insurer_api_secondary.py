# tools/insurer_api_secondary.py - Secondary Insurer API

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import asyncio
import pandas as pd
import random
from datetime import datetime, timedelta
import json

app = FastAPI(title="Secondary Insurance API - Cigna/United")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Secondary Insurance API"
    }

class ClaimSubmission(BaseModel):
    patient_id: str
    patient_name: str
    diagnosis: str
    icd_code: str
    cpt_code: str
    claim_amount: float
    insurance_company: str
    prior_auth: Optional[str] = None
    medical_history: Optional[str] = None
    provider_npi: str
    treatment_date: str
    x12_837p: Optional[str] = None   # Real X12 837P transaction

class AppealSubmission(BaseModel):
    claim_id: str
    appeal_reason: str
    supporting_documentation: str
    medical_necessity: str

# Store pending claims for delayed response
pending_claims: Dict[str, Dict[str, Any]] = {}

@app.post("/submit")
async def submit_claim(claim: ClaimSubmission):
    """Submit a claim — accepts X12 837P or structured fields"""

    claim_id = f"{claim.patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # ── X12 837P validation ──────────────────────────────────────────────
    x12_valid = False
    x12_summary = {}
    if claim.x12_837p:
        try:
            import sys, os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from tools.x12_837p_builder import parse_837p_summary
            x12_summary = parse_837p_summary(claim.x12_837p)
            x12_valid = bool(x12_summary.get("claim_id") and x12_summary.get("total_charge"))
            print(f"[837P] Valid={x12_valid} | Claim={x12_summary.get('claim_id')} | "
                  f"CPT={x12_summary.get('cpt_code')} | ICD={x12_summary.get('icd_code')} | "
                  f"Charge=${x12_summary.get('total_charge')}")
        except Exception as e:
            print(f"[837P] Parse error: {e}")

    approval_decision = determine_approval(claim)

    pending_claims[claim_id] = {
        "claim": claim.dict(),
        "x12_valid": x12_valid,
        "x12_summary": x12_summary,
        "decision": approval_decision,
        "submitted_at": datetime.now(),
        "processed": False
    }

    asyncio.create_task(process_claim_delayed(claim_id))

    return {
        "status": "pending",
        "claim_id": claim_id,
        "x12_received": bool(claim.x12_837p),
        "x12_valid": x12_valid,
        "message": "Claim submitted for processing. Result will be available in 60 seconds.",
        "estimated_processing_time": "60 seconds"
    }

@app.get("/claim-status/{claim_id}")
async def get_claim_status(claim_id: str):
    """Check the status of a submitted claim"""
    
    if claim_id not in pending_claims:
        raise HTTPException(status_code=404, detail="Claim not found")
    
    claim_info = pending_claims[claim_id]
    
    if not claim_info["processed"]:
        time_elapsed = (datetime.now() - claim_info["submitted_at"]).total_seconds()
        remaining_time = max(0, 180 - int(time_elapsed))
        
        return {
            "status": "processing",
            "claim_id": claim_id,
            "message": f"Claim is being processed. Check back in {remaining_time} seconds.",
            "time_remaining": remaining_time
        }
    
    return claim_info["decision"]

@app.post("/appeal")
async def submit_appeal(appeal: AppealSubmission):
    """Submit an appeal for a denied claim"""
    
    # Appeals have higher chance of approval
    appeal_success = random.random() < 0.55  # 55% success rate for appeals
    
    if appeal_success:
        return {
            "status": "approved",
            "claim_id": appeal.claim_id,
            "reason": "Appeal accepted - medical necessity established",
            "appeal_id": f"APP-{appeal.claim_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }
    else:
        return {
            "status": "denied",
            "claim_id": appeal.claim_id,
            "reason": "Appeal denied - treatment not covered under current plan",
            "appeal_id": f"APP-{appeal.claim_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }

async def process_claim_delayed(claim_id: str):
    """Process claim after delay"""
    await asyncio.sleep(60)  # Wait 60 seconds (1 minute)
    
    if claim_id in pending_claims:
        pending_claims[claim_id]["processed"] = True
        
        # Log the processing completion
        print(f"Claim {claim_id} processed after 60 seconds (1 minute)")

def determine_approval(claim: ClaimSubmission) -> Dict[str, Any]:
    """Determine claim approval based on realistic patterns - Different from Primary API"""
    
    # Load denial learning data to make intelligent decisions
    denial_patterns = load_denial_patterns()
    
    # Check for common denial reasons (different patterns than Primary API)
    denial_reason = None
    
    # Pattern 1: Network restrictions for Cigna/United
    if claim.insurance_company == "Cigna" and random.random() < 0.15:
        denial_reason = "Provider not in Cigna network - requires in-network referral"
    
    elif claim.insurance_company == "United" and claim.claim_amount > 350:
        denial_reason = "United requires pre-authorization for procedures over $350"
    
    # Pattern 2: Specific diagnosis restrictions
    elif "Chronic" in claim.diagnosis and not claim.medical_history:
        denial_reason = "Chronic conditions require detailed medical history documentation"
    
    # Pattern 3: Age-based restrictions
    elif claim.claim_amount > 500:
        denial_reason = "High-cost procedures require additional medical justification"
    
    # Pattern 4: Different random denial rate than Primary API
    elif random.random() < 0.20:  # 20% random denial rate
        denial_reasons = [
            "Experimental treatment not covered",
            "Exceeds annual benefit limit",
            "Requires specialist referral",
            "Treatment not medically necessary",
            "Missing lab results documentation"
        ]
        denial_reason = random.choice(denial_reasons)
    
    if denial_reason:
        return {
            "status": "denied",
            "claim_id": f"PENDING-{claim.patient_id}",
            "reason": denial_reason,
            "denial_code": f"D{random.randint(200, 899)}",
            "appeal_deadline": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        }
    else:
        return {
            "status": "approved",
            "claim_id": f"PENDING-{claim.patient_id}",
            "approval_amount": claim.claim_amount,
            "payment_date": (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
        }

def load_denial_patterns():
    """Load historical denial patterns for learning"""
    try:
        return pd.read_csv("data/denial_learning.csv")
    except:
        return pd.DataFrame()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)
