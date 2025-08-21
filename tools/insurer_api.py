# tools/insurer_api.py

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from random import random, choice
from typing import Optional
import uvicorn
from .denial_reasons import DenialReasonGenerator

app = FastAPI(title="Dummy Insurer API")

# ------------------------
# Claim Submission Endpoint
# ------------------------

class ClaimSubmission(BaseModel):
    patient_info: dict
    billing: dict
    diagnosis: dict
    metadata: Optional[dict] = {}

@app.post("/v1/submit")
async def submit_claim(claim: ClaimSubmission):
    denial_generator = DenialReasonGenerator()
    
    # Get patient info and claim details
    patient_name = claim.patient_info.get("name", "Unknown Patient")
    claim_amount = claim.billing.get("amount", 0)
    insurer = claim.metadata.get("insurer", "Unknown")
    
    # Simulate denial with specific reasons
    if random() < 0.3:  # 30% denial rate for demonstration
        denial_info = denial_generator.get_specific_denial(insurer, claim_amount)
        denial_message = denial_generator.format_denial_message(
            patient_name, claim_amount, insurer, denial_info
        )
        return {
            "status": "rejected",
            "message": denial_message,
            "denial_info": denial_info
        }

    return {"status": "approved"}

# ------------------------
# Appeal Resubmission Endpoint
# ------------------------

@app.post("/v1/appeal")
async def resubmit_appeal(file: UploadFile = File(...)):
    if file.filename.endswith(".pdf"):
        return {"status": choice(["approved", "rejected"])}
    return {"status": "rejected", "reason": "Appeal format not supported"}

# ------------------------
# Run the mock server
# ------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
