"""
Denial Classifier Service
Intelligently categorizes and analyzes denial reasons
"""

import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import re

class DenialClassifier:
    """Service for classifying and analyzing denial reasons"""
    
    def __init__(self):
        self.denial_taxonomy = self._load_denial_taxonomy()
        self.ml_confidence_threshold = 0.7
        
    def _load_denial_taxonomy(self) -> Dict[str, Any]:
        """Load denial reason taxonomy and classification rules"""
        
        return {
            "medical_necessity": {
                "keywords": ["medical necessity", "not medically necessary", "experimental", "investigational"],
                "codes": ["CO-50", "CO-96", "CO-151"],
                "category": "Medical Necessity",
                "appeal_strategy": "clinical_documentation",
                "success_rate": 0.65,
                "avg_processing_days": 14
            },
            "prior_authorization": {
                "keywords": ["prior authorization", "preauthorization", "precertification", "approval required"],
                "codes": ["CO-197", "CO-27"],
                "category": "Prior Authorization",
                "appeal_strategy": "authorization_request",
                "success_rate": 0.80,
                "avg_processing_days": 7
            },
            "documentation": {
                "keywords": ["documentation", "records", "information", "lacks information"],
                "codes": ["CO-16", "CO-29"],
                "category": "Insufficient Documentation",
                "appeal_strategy": "additional_documentation",
                "success_rate": 0.75,
                "avg_processing_days": 10
            },
            "coding_error": {
                "keywords": ["procedure code", "diagnosis code", "incorrect code", "invalid code"],
                "codes": ["CO-109", "CO-151"],
                "category": "Coding Error",
                "appeal_strategy": "code_correction",
                "success_rate": 0.85,
                "avg_processing_days": 5
            },
            "timely_filing": {
                "keywords": ["timely filing", "time limit", "filing deadline", "late submission"],
                "codes": ["CO-29", "CO-204"],
                "category": "Timely Filing",
                "appeal_strategy": "good_cause_explanation",
                "success_rate": 0.25,
                "avg_processing_days": 21
            },
            "policy_exclusion": {
                "keywords": ["not covered", "excluded", "benefit limitation", "policy exclusion"],
                "codes": ["CO-50", "CO-97", "CO-204"],
                "category": "Policy Exclusion",
                "appeal_strategy": "policy_interpretation",
                "success_rate": 0.40,
                "avg_processing_days": 18
            },
            "eligibility": {
                "keywords": ["not eligible", "coverage terminated", "inactive coverage"],
                "codes": ["CO-27", "CO-109"],
                "category": "Patient Eligibility",
                "appeal_strategy": "eligibility_verification",
                "success_rate": 0.30,
                "avg_processing_days": 12
            },
            "duplicate_claim": {
                "keywords": ["duplicate", "already paid", "previous payment"],
                "codes": ["CO-18", "CO-97"],
                "category": "Duplicate Claim",
                "appeal_strategy": "claim_differentiation",
                "success_rate": 0.70,
                "avg_processing_days": 8
            }
        }
    
    def classify_denial(self, denial_reason: str, denial_code: str = None) -> Dict[str, Any]:
        """Classify a denial reason into standardized categories"""
        
        # Primary classification based on denial code
        code_classification = self._classify_by_code(denial_code) if denial_code else None
        
        # Secondary classification based on text analysis
        text_classification = self._classify_by_text(denial_reason)
        
        # Combine classifications with confidence scoring
        final_classification = self._combine_classifications(
            code_classification, text_classification, denial_reason, denial_code
        )
        
        return final_classification
    
    def _classify_by_code(self, denial_code: str) -> Optional[Tuple[str, float]]:
        """Classify denial based on standard denial codes"""
        
        if not denial_code:
            return None
            
        for category_key, category_data in self.denial_taxonomy.items():
            if denial_code in category_data["codes"]:
                return (category_key, 0.9)  # High confidence for code-based classification
        
        return None
    
    def _classify_by_text(self, denial_reason: str) -> List[Tuple[str, float]]:
        """Classify denial based on text analysis"""
        
        denial_reason_lower = denial_reason.lower()
        matches = []
        
        for category_key, category_data in self.denial_taxonomy.items():
            confidence = 0.0
            keyword_matches = 0
            
            for keyword in category_data["keywords"]:
                if keyword.lower() in denial_reason_lower:
                    keyword_matches += 1
                    confidence += 0.2
            
            # Boost confidence for multiple keyword matches
            if keyword_matches > 1:
                confidence += 0.2
            
            if confidence > 0:
                matches.append((category_key, min(confidence, 1.0)))
        
        # Sort by confidence
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def _combine_classifications(self, code_class: Optional[Tuple[str, float]], 
                               text_classes: List[Tuple[str, float]], 
                               denial_reason: str, denial_code: str) -> Dict[str, Any]:
        """Combine code and text classifications into final result"""
        
        # Start with code classification if available (higher confidence)
        if code_class:
            primary_category = code_class[0]
            confidence = code_class[1]
            
            # Check if text classification agrees
            text_agreement = any(tc[0] == primary_category for tc in text_classes)
            if text_agreement:
                confidence = min(confidence + 0.1, 1.0)
        
        # Fall back to text classification
        elif text_classes:
            primary_category = text_classes[0][0]
            confidence = text_classes[0][1]
        
        # Default classification if nothing matches
        else:
            primary_category = "other"
            confidence = 0.3
        
        # Get category details
        category_info = self.denial_taxonomy.get(primary_category, {})
        
        # Generate alternative classifications
        alternatives = []
        if text_classes:
            for category, conf in text_classes[1:3]:  # Top 2 alternatives
                if category != primary_category:
                    alt_info = self.denial_taxonomy.get(category, {})
                    alternatives.append({
                        "category": category,
                        "confidence": conf,
                        "display_name": alt_info.get("category", category.replace("_", " ").title())
                    })
        
        return {
            "primary_classification": {
                "category": primary_category,
                "display_name": category_info.get("category", primary_category.replace("_", " ").title()),
                "confidence": round(confidence, 2)
            },
            "alternative_classifications": alternatives,
            "appeal_strategy": category_info.get("appeal_strategy", "general_appeal"),
            "expected_success_rate": category_info.get("success_rate", 0.50),
            "estimated_processing_days": category_info.get("avg_processing_days", 14),
            "classification_metadata": {
                "denial_reason": denial_reason,
                "denial_code": denial_code,
                "classified_at": datetime.now().isoformat(),
                "classification_method": "hybrid" if code_class and text_classes else 
                                       "code_based" if code_class else "text_based"
            }
        }
    
    def batch_classify_denials(self, denials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify multiple denials in batch"""
        
        classified_denials = []
        
        for denial in denials:
            denial_reason = denial.get("denial_reason", "")
            denial_code = denial.get("denial_code")
            
            classification = self.classify_denial(denial_reason, denial_code)
            
            # Enhance denial record with classification
            enhanced_denial = {
                **denial,
                "classification": classification,
                "recommended_actions": self._get_recommended_actions(classification),
                "priority_score": self._calculate_priority_score(denial, classification)
            }
            
            classified_denials.append(enhanced_denial)
        
        return classified_denials
    
    def _get_recommended_actions(self, classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommended actions based on classification"""
        
        strategy = classification.get("appeal_strategy", "general_appeal")
        category = classification["primary_classification"]["category"]
        
        action_templates = {
            "clinical_documentation": [
                {
                    "action": "Gather comprehensive medical records",
                    "priority": "high",
                    "effort": "medium",
                    "description": "Collect all relevant clinical documentation supporting medical necessity"
                },
                {
                    "action": "Request physician peer-to-peer review",
                    "priority": "medium", 
                    "effort": "low",
                    "description": "Schedule call between reviewing physician and treating physician"
                },
                {
                    "action": "Submit clinical guidelines and evidence",
                    "priority": "high",
                    "effort": "medium",
                    "description": "Reference published guidelines supporting the treatment approach"
                }
            ],
            "authorization_request": [
                {
                    "action": "Submit retroactive prior authorization",
                    "priority": "high",
                    "effort": "low",
                    "description": "Request authorization with clinical justification"
                },
                {
                    "action": "Document emergency/urgent nature",
                    "priority": "medium",
                    "effort": "low", 
                    "description": "Provide evidence that service could not wait for authorization"
                }
            ],
            "additional_documentation": [
                {
                    "action": "Submit missing documentation",
                    "priority": "high",
                    "effort": "low",
                    "description": "Provide all requested information and documentation"
                },
                {
                    "action": "Verify claim completeness",
                    "priority": "medium",
                    "effort": "low",
                    "description": "Double-check all required fields are completed"
                }
            ],
            "code_correction": [
                {
                    "action": "Review and correct procedure codes",
                    "priority": "high",
                    "effort": "low",
                    "description": "Verify CPT codes match documented procedures"
                },
                {
                    "action": "Validate diagnosis code linkage",
                    "priority": "high", 
                    "effort": "low",
                    "description": "Ensure ICD codes support the procedures performed"
                }
            ]
        }
        
        return action_templates.get(strategy, [
            {
                "action": "Review denial and gather evidence",
                "priority": "medium",
                "effort": "medium",
                "description": "Analyze denial reason and collect supporting documentation"
            }
        ])
    
    def _calculate_priority_score(self, denial: Dict[str, Any], 
                                classification: Dict[str, Any]) -> float:
        """Calculate priority score for denial appeal"""
        
        base_score = 0.5
        
        # Amount factor (higher amounts = higher priority)
        amount = denial.get("denied_amount", 0)
        if amount > 5000:
            base_score += 0.3
        elif amount > 2000:
            base_score += 0.2
        elif amount > 1000:
            base_score += 0.1
        
        # Success rate factor
        success_rate = classification.get("expected_success_rate", 0.5)
        base_score += (success_rate - 0.5) * 0.4
        
        # Processing time factor (faster = slightly higher priority)
        processing_days = classification.get("estimated_processing_days", 14)
        if processing_days <= 7:
            base_score += 0.1
        elif processing_days > 21:
            base_score -= 0.1
        
        # Classification confidence factor
        confidence = classification["primary_classification"]["confidence"]
        base_score += (confidence - 0.5) * 0.2
        
        return round(max(0.0, min(1.0, base_score)), 2)
    
    def get_classification_analytics(self, classified_denials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate analytics from classified denials"""
        
        if not classified_denials:
            return {"error": "No classified denials provided"}
        
        # Count by category
        category_counts = {}
        total_amount_by_category = {}
        confidence_scores = []
        priority_scores = []
        
        for denial in classified_denials:
            classification = denial.get("classification", {})
            category = classification["primary_classification"]["category"]
            
            category_counts[category] = category_counts.get(category, 0) + 1
            
            amount = denial.get("denied_amount", 0)
            total_amount_by_category[category] = total_amount_by_category.get(category, 0) + amount
            
            confidence_scores.append(classification["primary_classification"]["confidence"])
            priority_scores.append(denial.get("priority_score", 0.5))
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        avg_priority = sum(priority_scores) / len(priority_scores)
        
        return {
            "total_denials_classified": len(classified_denials),
            "category_distribution": category_counts,
            "amount_by_category": {k: round(v, 2) for k, v in total_amount_by_category.items()},
            "average_classification_confidence": round(avg_confidence, 3),
            "average_priority_score": round(avg_priority, 3),
            "high_priority_denials": len([d for d in classified_denials if d.get("priority_score", 0) > 0.7]),
            "analytics_generated_at": datetime.now().isoformat()
        }
