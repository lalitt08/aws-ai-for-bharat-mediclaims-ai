from typing import Dict, List
import random
from datetime import datetime

class DenialReasonGenerator:
    def __init__(self):
        self.denial_patterns = {
            "United": {
                "documentation": [
                    {
                        "reason": "Missing clinical documentation",
                        "details": "Required: Progress notes from last 3 visits, vital signs, and treatment plan",
                        "required_items": [
                            "Progress notes from previous visits",
                            "Current vital signs",
                            "Detailed treatment plan"
                        ],
                        "success_rate": 0.85
                    },
                    {
                        "reason": "Incomplete provider credentials",
                        "details": "Provider NPI validation failed. Update provider enrollment status",
                        "required_items": [
                            "Updated NPI documentation",
                            "Provider enrollment verification",
                            "State license information"
                        ],
                        "success_rate": 0.90
                    }
                ]
            },
            "Aetna": {
                "coding": [
                    {
                        "reason": "Diagnosis code mismatch",
                        "details": "Primary diagnosis code inconsistent with procedure performed",
                        "required_items": [
                            "Updated ICD-10 codes",
                            "Clinical documentation supporting diagnosis",
                            "Medical necessity justification"
                        ],
                        "success_rate": 0.75
                    },
                    {
                        "reason": "Modifier usage error",
                        "details": "Incorrect modifier used for bilateral procedure",
                        "required_items": [
                            "Correct procedure modifiers",
                            "Operative notes",
                            "Anatomical identification"
                        ],
                        "success_rate": 0.80
                    }
                ]
            },
            "BlueCross": {
                "authorization": [
                    {
                        "reason": "Prior authorization expired",
                        "details": "Authorization expired before service date. Renewal required",
                        "required_items": [
                            "New prior authorization",
                            "Clinical necessity documentation",
                            "Updated service dates"
                        ],
                        "success_rate": 0.70
                    },
                    {
                        "reason": "Service level mismatch",
                        "details": "Authorized service level differs from provided service",
                        "required_items": [
                            "Updated authorization",
                            "Service level documentation",
                            "Clinical justification for level of care"
                        ],
                        "success_rate": 0.75
                    }
                ]
            },
            "Cigna": {
                "medical_necessity": [
                    {
                        "reason": "Insufficient medical necessity documentation",
                        "details": "Documentation does not support medical necessity for procedure",
                        "required_items": [
                            "Detailed clinical findings",
                            "Conservative treatment history",
                            "Objective test results"
                        ],
                        "success_rate": 0.80
                    },
                    {
                        "reason": "Missing treatment history",
                        "details": "Prior treatment attempts not documented",
                        "required_items": [
                            "Previous treatment records",
                            "Response to prior treatments",
                            "Justification for current treatment plan"
                        ],
                        "success_rate": 0.85
                    }
                ]
            }
        }

    def get_specific_denial(self, insurer: str, claim_amount: float) -> Dict:
        """Generate a specific, meaningful denial reason based on insurer and claim details"""
        
        if insurer not in self.denial_patterns:
            return self._generate_generic_denial()

        # Select a random category for the insurer
        category = random.choice(list(self.denial_patterns[insurer].keys()))
        
        # Select a specific denial pattern from the category
        denial_pattern = random.choice(self.denial_patterns[insurer][category])

        return {
            "status": "REJECTED",
            "reason": denial_pattern["reason"],
            "details": denial_pattern["details"],
            "required_items": denial_pattern["required_items"],
            "estimated_success_rate": denial_pattern["success_rate"],
            "next_steps": "Please provide the required documentation for resubmission"
        }

    def _generate_generic_denial(self) -> Dict:
        """Fallback generic denial reason"""
        return {
            "status": "REJECTED",
            "reason": "Documentation requirements not met",
            "details": "Please review submission guidelines",
            "required_items": ["Complete medical records", "Updated claim form"],
            "estimated_success_rate": 0.60,
            "next_steps": "Contact provider services for specific requirements"
        }

    def format_denial_message(self, patient_name: str, claim_amount: float, 
                            insurer: str, denial_info: Dict) -> str:
        """Format a user-friendly denial message"""
        
        current_time = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
        
        message = f"{patient_name} REJECTED\n"
        message += f"Claim Amount: ${claim_amount}\n"
        message += f"Insurer: {insurer}\n"
        message += f"Rejected: {current_time}\n\n"
        message += f"Reason: {denial_info['reason']}\n"
        message += f"Details: {denial_info['details']}\n\n"
        message += "Required for Resubmission:\n"
        for item in denial_info['required_items']:
            message += f"- {item}\n"
        message += f"\nEstimated approval rate after providing required items: {denial_info['success_rate']*100}%\n"
        
        return message
