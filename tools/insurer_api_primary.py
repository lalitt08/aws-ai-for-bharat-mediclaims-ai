# tools/insurer_api_primary.py - Primary Insurer API

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import asyncio
import pandas as pd
import random
from datetime import datetime, timedelta
import json

app = FastAPI(title="Primary Insurance API - BlueCross/Aetna")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Primary Insurance API"
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

# Import denial patterns handler with absolute import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.denial_patterns_handler import DenialPatternsHandler

denial_handler = DenialPatternsHandler()

@app.post("/submit")
async def submit_claim(claim: ClaimSubmission):
    """Submit a claim for processing — accepts X12 837P or structured fields"""

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
            # Basic validation: must have CLM segment with claim_id and charge
            x12_valid = bool(x12_summary.get("claim_id") and x12_summary.get("total_charge"))
            print(f"[837P] Valid={x12_valid} | Claim={x12_summary.get('claim_id')} | "
                  f"Patient={x12_summary.get('patient_last')},{x12_summary.get('patient_first')} | "
                  f"CPT={x12_summary.get('cpt_code')} | ICD={x12_summary.get('icd_code')} | "
                  f"Charge=${x12_summary.get('total_charge')}")
        except Exception as e:
            print(f"[837P] Parse error: {e}")

    # ── Adjudication logic ───────────────────────────────────────────────
    if random.random() < 0.3:
        denial_result = denial_handler.get_denial_message(
            patient_name=claim.patient_name,
            claim_amount=claim.claim_amount,
            insurance_company=claim.insurance_company
        )
        decision = denial_result
    else:
        decision = {
            "status": "approved",
            "approved_amount": claim.claim_amount * random.uniform(0.85, 1.0),
            "message": f"Claim approved for ${claim.claim_amount:.2f}"
        }

    pending_claims[claim_id] = {
        "claim": claim.dict(),
        "x12_valid": x12_valid,
        "x12_summary": x12_summary,
        "decision": decision,
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
    appeal_success = random.random() < 0.6  # 60% success rate for appeals
    
    if appeal_success:
        return {
            "status": "approved",
            "claim_id": appeal.claim_id,
            "reason": "Appeal accepted based on additional documentation",
            "appeal_id": f"APP-{appeal.claim_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }
    else:
        return {
            "status": "denied",
            "claim_id": appeal.claim_id,
            "reason": "Appeal denied - insufficient medical necessity documentation",
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
    """Determine claim approval based on realistic patterns"""
    
    # Load denial learning data to make intelligent decisions
    denial_patterns = load_denial_patterns()
    
    # Check for common denial reasons
    denial_reason = None
    
    # Pattern 1: Missing prior authorization for expensive procedures
    if claim.claim_amount > 300 and not claim.prior_auth:
        denial_reason = "Missing prior authorization for high-cost procedure"
    
    # Pattern 2: Invalid CPT/ICD combinations
    elif is_invalid_code_combination(claim.icd_code, claim.cpt_code):
        denial_reason = "CPT code does not match diagnosis (ICD code)"
    
    # Pattern 3: Insurance-specific patterns
    elif claim.insurance_company == "BlueCross" and "Diabetes" in claim.diagnosis and not claim.prior_auth:
        denial_reason = "BlueCross requires prior authorization for diabetes management"
    
    elif claim.insurance_company == "Aetna" and claim.claim_amount > 400:
        denial_reason = "Aetna requires additional documentation for claims over $400"
    
    # Pattern 4: Random denials based on historical patterns
    elif random.random() < 0.25:  # 25% random denial rate
        denial_reasons = [
            "Insufficient medical documentation",
            "Procedure not medically necessary",
            "Pre-existing condition exclusion",
            "Provider not in network",
            "Duplicate claim submission"
        ]
        denial_reason = random.choice(denial_reasons)
    
    if denial_reason:
        return {
            "status": "denied",
            "claim_id": f"PENDING-{claim.patient_id}",
            "reason": denial_reason,
            "denial_code": f"D{random.randint(100, 999)}",
            "appeal_deadline": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        }
    else:
        return {
            "status": "approved",
            "claim_id": f"PENDING-{claim.patient_id}",
            "approval_amount": claim.claim_amount,
            "payment_date": (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
        }

def load_denial_patterns():
    """Load historical denial patterns for learning"""
    try:
        return pd.read_csv("data/denial_learning.csv")
    except:
        return pd.DataFrame()

def is_invalid_code_combination(icd_code: str, cpt_code: str) -> bool:
    """Check if ICD and CPT codes are valid combination"""
    # Simplified validation - in real system this would be comprehensive
    invalid_combinations = [
        ("J30.9", "99215"),  # Allergic rhinitis shouldn't need complex exam
        ("G43.9", "99215"),  # Migraine shouldn't need complex exam
        ("F41.9", "94640"),  # Anxiety disorder with pulmonary function test
    ]
    
    return (icd_code, cpt_code) in invalid_combinations

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
