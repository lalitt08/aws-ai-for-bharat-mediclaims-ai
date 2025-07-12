# tools/insurer_api.py

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from random import random, choice
from typing import Optional
import uvicorn

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
    # Simulate denial if prior_auth is missing or random chance
    prior_auth = claim.billing.get("prior_auth")
    codes = claim.billing.get("codes", [])

    if not prior_auth or "XYZ123" in codes:
        return {"status": "rejected", "reason": "Missing or invalid prior_auth / CPT code"}

    if random() < 0.1:
        return {"status": "rejected", "reason": "Random rejection (for realism)"}

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
