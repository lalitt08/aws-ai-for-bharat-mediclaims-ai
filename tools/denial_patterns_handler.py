# tools/denial_patterns_handler.py

from typing import Dict, Any
from datetime import datetime
import random

class DenialPatternsHandler:
    def __init__(self):
        self.denial_patterns = {
            "BlueCross": [
                {
                    "reason": "Prior Authorization Expired",
                    "details": "Authorization expired or not valid for service date",
                    "requirements": [
                        "Updated prior authorization form",
                        "Clinical documentation supporting medical necessity",
                        "Updated service dates"
                    ],
                    "success_rate": 0.75
                },
                {
                    "reason": "Service Level Mismatch",
                    "details": "Billed service level exceeds authorized level of care",
                    "requirements": [
                        "Updated level of care documentation",
                        "Clinical justification for service level",
                        "Provider credentials for service level"
                    ],
                    "success_rate": 0.80
                }
            ],
            "Aetna": [
                {
                    "reason": "Diagnosis Code Specificity",
                    "details": "ICD-10 code requires higher specificity for procedure",
                    "requirements": [
                        "Updated diagnosis codes",
                        "Clinical notes supporting diagnosis",
                        "Recent examination findings"
                    ],
                    "success_rate": 0.85
                },
                {
                    "reason": "Medical Record Documentation",
                    "details": "Insufficient clinical documentation for service provided",
                    "requirements": [
                        "Complete progress notes",
                        "Relevant test results",
                        "Treatment plan documentation"
                    ],
                    "success_rate": 0.90
                }
            ],
            "Cigna": [
                {
                    "reason": "Medical Necessity Criteria",
                    "details": "Documentation does not meet medical necessity guidelines",
                    "requirements": [
                        "Clinical findings documentation",
                        "Failed conservative treatment history",
                        "Objective measurement data"
                    ],
                    "success_rate": 0.70
                },
                {
                    "reason": "Treatment Plan Documentation",
                    "details": "Incomplete or missing treatment plan documentation",
                    "requirements": [
                        "Detailed treatment goals",
                        "Expected outcomes",
                        "Treatment frequency and duration"
                    ],
                    "success_rate": 0.85
                }
            ],
            "United": [
                {
                    "reason": "Provider Network Status",
                    "details": "Provider credentials require verification for service",
                    "requirements": [
                        "Updated provider credentialing",
                        "Network participation verification",
                        "Facility accreditation documentation"
                    ],
                    "success_rate": 0.80
                },
                {
                    "reason": "Procedure Code Documentation",
                    "details": "Procedure documentation incomplete for billed code",
                    "requirements": [
                        "Complete procedure notes",
                        "Supporting clinical indicators",
                        "Time documentation for timed codes"
                    ],
                    "success_rate": 0.85
                }
            ]
        }

    def get_denial_message(self, patient_name: str, claim_amount: float, 
                          insurance_company: str) -> Dict[str, Any]:
        """Generate a specific denial message with detailed requirements"""
        
        if insurance_company not in self.denial_patterns:
            return self._generate_generic_denial(patient_name, claim_amount, insurance_company)

        # Select a random denial pattern for the insurance company
        denial_pattern = random.choice(self.denial_patterns[insurance_company])
        
        current_time = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
        
        message = (
            f"{patient_name} REJECTED\n"
            f"Claim Amount: ${claim_amount}\n"
            f"Insurer: {insurance_company}\n"
            f"Rejected: {current_time}\n\n"
            f"Reason: {denial_pattern['reason']}\n"
            f"Details: {denial_pattern['details']}\n\n"
            f"Required for Resubmission:\n"
        )
        
        for requirement in denial_pattern['requirements']:
            message += f"- {requirement}\n"
            
        message += f"\nEstimated approval rate after providing required items: {denial_pattern['success_rate']*100}%"
        
        return {
            "status": "rejected",
            "message": message,
            "reason": denial_pattern['reason'],
            "requirements": denial_pattern['requirements'],
            "success_rate": denial_pattern['success_rate']
        }

    def _generate_generic_denial(self, patient_name: str, claim_amount: float, 
                               insurance_company: str) -> Dict[str, Any]:
        """Generate a generic denial message for unknown insurers"""
        
        current_time = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
        
        message = (
            f"{patient_name} REJECTED\n"
            f"Claim Amount: ${claim_amount}\n"
            f"Insurer: {insurance_company}\n"
            f"Rejected: {current_time}\n\n"
            f"Reason: Documentation Requirements Not Met\n"
            f"Details: Additional documentation needed for claim processing\n\n"
            f"Required for Resubmission:\n"
            "- Complete medical records\n"
            "- Updated claim form\n"
            "- Supporting clinical documentation\n"
        )
        
        return {
            "status": "rejected",
            "message": message,
            "reason": "Documentation Requirements Not Met",
            "requirements": [
                "Complete medical records",
                "Updated claim form",
                "Supporting clinical documentation"
            ],
            "success_rate": 0.60
        }
