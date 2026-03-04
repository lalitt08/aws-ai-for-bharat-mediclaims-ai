"""
Appeal Generator Service
Creates intelligent appeals based on denial analysis
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

class AppealGenerator:
    """Service for generating intelligent appeals based on denial analysis"""
    
    def __init__(self):
        self.appeal_templates = self._load_appeal_templates()
        self.clinical_justification_library = self._load_clinical_justifications()
        
    def _load_appeal_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load appeal letter templates for different denial types"""
        
        return {
            "medical_necessity": {
                "opening": "We respectfully appeal the denial of the above-referenced claim based on medical necessity. The services provided were medically necessary and appropriate for the patient's condition.",
                "evidence_section": "Medical Evidence Supporting Necessity",
                "closing": "Based on the clinical evidence presented, we request reconsideration and approval of this claim.",
                "key_points": [
                    "Patient's clinical presentation and symptoms",
                    "Conservative treatment attempts and outcomes",
                    "Medical literature supporting intervention",
                    "Provider's clinical judgment and experience"
                ]
            },
            "prior_authorization": {
                "opening": "We are appealing the denial due to lack of prior authorization. We believe this denial should be overturned due to the following circumstances:",
                "evidence_section": "Justification for Retroactive Authorization",
                "closing": "We request retroactive authorization and payment for this medically necessary service.",
                "key_points": [
                    "Emergency or urgent nature of service",
                    "Clinical circumstances preventing prior authorization",
                    "Standard of care requirements",
                    "Patient safety considerations"
                ]
            },
            "documentation": {
                "opening": "We are submitting additional documentation to support the medical necessity and appropriateness of the services rendered.",
                "evidence_section": "Supporting Documentation",
                "closing": "With this additional documentation, we request reconsideration and approval of this claim.",
                "key_points": [
                    "Complete medical records",
                    "Provider notes and assessments",
                    "Diagnostic test results",
                    "Treatment plans and outcomes"
                ]
            },
            "coding_error": {
                "opening": "We are appealing the denial and submitting corrected coding information. The original claim contained coding errors that have been identified and corrected.",
                "evidence_section": "Corrected Coding Information",
                "closing": "We request processing of the claim with the corrected codes provided.",
                "key_points": [
                    "Identification of coding errors",
                    "Corrected procedure codes",
                    "Updated diagnosis codes",
                    "Code linkage justification"
                ]
            }
        }
    
    def _load_clinical_justifications(self) -> Dict[str, List[str]]:
        """Load clinical justification statements by category"""
        
        return {
            "medical_necessity": [
                "The patient's condition required immediate intervention to prevent deterioration",
                "Conservative treatment options were exhausted or contraindicated",
                "The procedure follows established clinical guidelines and standards of care",
                "The service was the least invasive option available for the patient's condition",
                "Delaying treatment would have resulted in significant patient morbidity"
            ],
            "emergency_services": [
                "The patient presented with acute symptoms requiring immediate attention",
                "The condition met emergency department criteria per EMTALA guidelines",
                "Stabilization was required before transfer or discharge",
                "The situation did not allow time for prior authorization procedures"
            ],
            "specialist_referral": [
                "The complexity of the condition required specialist expertise",
                "Primary care management was insufficient for the patient's needs",
                "Specialist intervention was necessary for accurate diagnosis",
                "The referral followed appropriate clinical protocols"
            ]
        }
    
    def generate_appeal(self, denial_data: Dict[str, Any], 
                       classification: Dict[str, Any] = None,
                       additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a comprehensive appeal based on denial analysis"""
        
        # Determine appeal strategy
        strategy = self._determine_appeal_strategy(denial_data, classification)
        
        # Generate appeal content
        appeal_content = self._generate_appeal_content(denial_data, strategy, additional_context)
        
        # Generate supporting documentation requirements
        documentation_requirements = self._generate_documentation_requirements(strategy)
        
        # Calculate appeal metadata
        metadata = self._calculate_appeal_metadata(denial_data, classification, strategy)
        
        appeal_id = f"APP-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return {
            "appeal_id": appeal_id,
            "claim_id": denial_data.get("claim_id"),
            "denial_code": denial_data.get("denial_code"),
            "appeal_strategy": strategy,
            "appeal_content": appeal_content,
            "documentation_requirements": documentation_requirements,
            "metadata": metadata,
            "generated_at": datetime.now().isoformat(),
            "status": "draft",
            "estimated_completion_time": self._estimate_completion_time(strategy),
            "success_probability": self._calculate_success_probability(denial_data, classification, strategy)
        }
    
    def _determine_appeal_strategy(self, denial_data: Dict[str, Any], 
                                 classification: Dict[str, Any] = None) -> str:
        """Determine the best appeal strategy based on denial analysis"""
        
        if classification:
            return classification.get("appeal_strategy", "general_appeal")
        
        # Fallback to denial code analysis
        denial_code = denial_data.get("denial_code", "")
        denial_reason = denial_data.get("denial_reason", "").lower()
        
        if "medical necessity" in denial_reason or denial_code in ["CO-50", "CO-96"]:
            return "medical_necessity"
        elif "prior authorization" in denial_reason or denial_code == "CO-197":
            return "prior_authorization"
        elif "documentation" in denial_reason or denial_code == "CO-16":
            return "documentation"
        elif "code" in denial_reason or denial_code in ["CO-109", "CO-151"]:
            return "coding_error"
        else:
            return "general_appeal"
    
    def _generate_appeal_content(self, denial_data: Dict[str, Any], 
                               strategy: str, additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate the main appeal letter content"""
        
        template = self.appeal_templates.get(strategy, self.appeal_templates["medical_necessity"])
        
        # Generate header information
        header = self._generate_appeal_header(denial_data)
        
        # Generate main content sections
        opening_paragraph = template["opening"]
        
        # Generate evidence section
        evidence_points = self._generate_evidence_points(denial_data, strategy, additional_context)
        
        # Generate clinical justification
        clinical_justification = self._generate_clinical_justification(denial_data, strategy)
        
        # Generate closing
        closing_paragraph = template["closing"]
        
        return {
            "header": header,
            "opening": opening_paragraph,
            "evidence_section": {
                "title": template["evidence_section"],
                "points": evidence_points
            },
            "clinical_justification": clinical_justification,
            "closing": closing_paragraph,
            "signature_block": self._generate_signature_block(),
            "attachments_list": self._generate_attachments_list(strategy)
        }
    
    def _generate_appeal_header(self, denial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appeal letter header information"""
        
        return {
            "date": datetime.now().strftime("%B %d, %Y"),
            "to": f"{denial_data.get('payer', 'Insurance Company')} Appeals Department",
            "re": {
                "patient_name": denial_data.get("patient_name", "Patient"),
                "claim_number": denial_data.get("claim_id"),
                "service_date": denial_data.get("service_date"),
                "provider": denial_data.get("provider_name", "Healthcare Provider"),
                "denied_amount": denial_data.get("denied_amount")
            }
        }
    
    def _generate_evidence_points(self, denial_data: Dict[str, Any], 
                                strategy: str, additional_context: Dict[str, Any] = None) -> List[str]:
        """Generate specific evidence points for the appeal"""
        
        base_points = []
        
        if strategy == "medical_necessity":
            base_points = [
                f"The patient's diagnosis of {denial_data.get('diagnosis', 'condition')} required the specific intervention provided",
                "The treatment followed established clinical guidelines and standards of care",
                "Conservative treatment options were considered and deemed inappropriate or insufficient",
                "The procedure was performed by a qualified specialist with appropriate expertise"
            ]
        elif strategy == "prior_authorization":
            base_points = [
                "The service was provided under emergency/urgent circumstances",
                "Prior authorization was not feasible due to the acute nature of the condition",
                "The treatment was medically necessary and could not be delayed",
                "The provider followed all appropriate emergency protocols"
            ]
        elif strategy == "documentation":
            base_points = [
                "Complete medical records are attached supporting the service provided",
                "All required documentation has been reviewed and verified",
                "Additional clinical notes provide further justification",
                "The documentation clearly demonstrates medical necessity"
            ]
        elif strategy == "coding_error":
            base_points = [
                "The original coding error has been identified and corrected",
                "The corrected codes accurately reflect the services performed",
                "The diagnosis codes support the procedure codes",
                "The claim should be reprocessed with the corrected information"
            ]
        
        # Add context-specific points if provided
        if additional_context:
            context_points = additional_context.get("evidence_points", [])
            base_points.extend(context_points)
        
        return base_points
    
    def _generate_clinical_justification(self, denial_data: Dict[str, Any], strategy: str) -> str:
        """Generate clinical justification narrative"""
        
        justifications = self.clinical_justification_library.get(strategy, 
                        self.clinical_justification_library.get("medical_necessity", []))
        
        if justifications:
            selected = random.choice(justifications)
            return f"{selected} The clinical documentation supports this determination and demonstrates compliance with accepted medical standards."
        
        return "The services provided were clinically appropriate and medically necessary based on the patient's condition and established standards of care."
    
    def _generate_documentation_requirements(self, strategy: str) -> List[Dict[str, Any]]:
        """Generate list of required supporting documentation"""
        
        requirements_map = {
            "medical_necessity": [
                {"document": "Complete medical records", "required": True, "description": "All relevant clinical documentation"},
                {"document": "Provider clinical notes", "required": True, "description": "Detailed notes supporting medical necessity"},
                {"document": "Clinical guidelines", "required": False, "description": "Published guidelines supporting treatment"},
                {"document": "Peer review documentation", "required": False, "description": "Independent physician review if available"}
            ],
            "prior_authorization": [
                {"document": "Emergency department records", "required": True, "description": "Documentation of emergency presentation"},
                {"document": "Prior authorization request", "required": True, "description": "Retroactive authorization request form"},
                {"document": "Clinical justification letter", "required": True, "description": "Provider letter explaining circumstances"},
                {"document": "Hospital admission records", "required": False, "description": "If applicable to the case"}
            ],
            "documentation": [
                {"document": "Additional medical records", "required": True, "description": "Previously missing documentation"},
                {"document": "Test results and imaging", "required": True, "description": "All diagnostic studies"},
                {"document": "Treatment plans", "required": True, "description": "Documented treatment approach"},
                {"document": "Progress notes", "required": False, "description": "Follow-up documentation"}
            ],
            "coding_error": [
                {"document": "Corrected claim form", "required": True, "description": "Claim with corrected codes"},
                {"document": "Coding justification", "required": True, "description": "Explanation of code corrections"},
                {"document": "Medical records", "required": True, "description": "Supporting documentation for codes"},
                {"document": "Coding guidelines reference", "required": False, "description": "Official coding guidance"}
            ]
        }
        
        return requirements_map.get(strategy, requirements_map["medical_necessity"])
    
    def _calculate_appeal_metadata(self, denial_data: Dict[str, Any], 
                                 classification: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """Calculate appeal metadata and predictions"""
        
        # Base metadata
        metadata = {
            "appeal_type": strategy,
            "denial_category": classification.get("primary_classification", {}).get("category") if classification else "unknown",
            "claim_amount": denial_data.get("denied_amount", 0),
            "complexity_score": self._calculate_complexity_score(denial_data, strategy),
            "required_effort": self._estimate_required_effort(strategy),
            "recommended_timeline": self._get_recommended_timeline(strategy)
        }
        
        return metadata
    
    def _calculate_complexity_score(self, denial_data: Dict[str, Any], strategy: str) -> float:
        """Calculate appeal complexity score (0.0 = simple, 1.0 = very complex)"""
        
        base_score = 0.5
        
        # Strategy-based complexity
        strategy_complexity = {
            "coding_error": 0.2,
            "documentation": 0.4,
            "prior_authorization": 0.6,
            "medical_necessity": 0.8
        }
        
        base_score = strategy_complexity.get(strategy, 0.5)
        
        # Amount-based adjustment
        amount = denial_data.get("denied_amount", 0)
        if amount > 10000:
            base_score += 0.2
        elif amount > 5000:
            base_score += 0.1
        
        return round(min(1.0, base_score), 2)
    
    def _estimate_required_effort(self, strategy: str) -> str:
        """Estimate effort required for appeal"""
        
        effort_map = {
            "coding_error": "Low",
            "documentation": "Medium", 
            "prior_authorization": "Medium",
            "medical_necessity": "High"
        }
        
        return effort_map.get(strategy, "Medium")
    
    def _get_recommended_timeline(self, strategy: str) -> Dict[str, Any]:
        """Get recommended timeline for appeal completion"""
        
        timeline_map = {
            "coding_error": {"preparation_days": 2, "submission_days": 1, "total_days": 3},
            "documentation": {"preparation_days": 5, "submission_days": 2, "total_days": 7},
            "prior_authorization": {"preparation_days": 7, "submission_days": 3, "total_days": 10},
            "medical_necessity": {"preparation_days": 14, "submission_days": 5, "total_days": 19}
        }
        
        return timeline_map.get(strategy, {"preparation_days": 7, "submission_days": 3, "total_days": 10})
    
    def _estimate_completion_time(self, strategy: str) -> str:
        """Estimate time to complete appeal preparation"""
        
        timeline = self._get_recommended_timeline(strategy)
        total_days = timeline["total_days"]
        
        if total_days <= 3:
            return "1-3 business days"
        elif total_days <= 7:
            return "3-7 business days"
        elif total_days <= 14:
            return "1-2 weeks"
        else:
            return "2-3 weeks"
    
    def _calculate_success_probability(self, denial_data: Dict[str, Any], 
                                     classification: Dict[str, Any], strategy: str) -> float:
        """Calculate probability of appeal success"""
        
        # Base probability from classification
        base_prob = 0.50
        if classification:
            base_prob = classification.get("expected_success_rate", 0.50)
        
        # Strategy-specific adjustments
        strategy_adjustments = {
            "coding_error": 0.15,      # Usually successful if correct
            "documentation": 0.10,     # Good if documentation is complete
            "prior_authorization": 0.05, # Moderate improvement
            "medical_necessity": -0.05  # Slightly more challenging
        }
        
        adjusted_prob = base_prob + strategy_adjustments.get(strategy, 0)
        
        # Amount-based adjustment (higher amounts get more attention)
        amount = denial_data.get("denied_amount", 0)
        if amount > 5000:
            adjusted_prob += 0.05
        elif amount < 500:
            adjusted_prob -= 0.05
        
        return round(max(0.1, min(0.95, adjusted_prob)), 2)
    
    def _generate_signature_block(self) -> Dict[str, str]:
        """Generate signature block for appeal letter"""
        
        return {
            "closing": "Sincerely,",
            "name": "Appeals Specialist",
            "title": "Revenue Cycle Management",
            "organization": "Healthcare Provider",
            "contact": "Phone: (555) 123-4567 | Email: appeals@provider.com"
        }
    
    def _generate_attachments_list(self, strategy: str) -> List[str]:
        """Generate list of standard attachments for appeal type"""
        
        attachments_map = {
            "medical_necessity": [
                "Medical records and clinical notes",
                "Provider documentation of medical necessity",
                "Clinical guidelines or literature references"
            ],
            "prior_authorization": [
                "Retroactive prior authorization request",
                "Emergency treatment documentation",
                "Clinical justification for urgency"
            ],
            "documentation": [
                "Complete medical records",
                "Additional clinical documentation",
                "Provider attestation of services"
            ],
            "coding_error": [
                "Corrected claim with proper codes",
                "Coding reference documentation",
                "Medical records supporting corrected codes"
            ]
        }
        
        return attachments_map.get(strategy, ["Supporting medical documentation", "Appeal justification letter"])
