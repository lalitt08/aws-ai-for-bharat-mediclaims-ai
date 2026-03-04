"""
Compliance Checker Service
Validates appeals and claims against regulatory requirements
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

class ComplianceChecker:
    """Service for checking regulatory compliance of claims and appeals"""
    
    def __init__(self):
        self.hipaa_rules = self._load_hipaa_rules()
        self.state_regulations = self._load_state_regulations()
        self.payer_policies = self._load_payer_policies()
        self.timely_filing_rules = self._load_timely_filing_rules()
        
    def _load_hipaa_rules(self) -> Dict[str, Any]:
        """Load HIPAA compliance rules"""
        
        return {
            "patient_consent": {
                "required": True,
                "description": "Patient consent required for PHI disclosure",
                "regulation": "45 CFR § 164.508",
                "severity": "high"
            },
            "minimum_necessary": {
                "required": True,
                "description": "Only minimum necessary PHI should be disclosed",
                "regulation": "45 CFR § 164.502(b)",
                "severity": "medium"
            },
            "audit_trail": {
                "required": True,
                "description": "Maintain audit trail of PHI access and disclosure",
                "regulation": "45 CFR § 164.312(b)",
                "severity": "medium"
            },
            "data_retention": {
                "required": True,
                "description": "Proper data retention and disposal procedures",
                "regulation": "45 CFR § 164.316(b)(2)",
                "severity": "low"
            }
        }
    
    def _load_state_regulations(self) -> Dict[str, Dict[str, Any]]:
        """Load state-specific regulations"""
        
        return {
            "CA": {
                "patient_rights": {
                    "description": "Enhanced patient rights and consent requirements",
                    "requirements": ["Written consent in primary language", "Informed consent documentation"],
                    "penalty": "high"
                },
                "network_adequacy": {
                    "description": "Provider network adequacy requirements",
                    "requirements": ["Adequate provider access", "Geographic coverage"],
                    "penalty": "medium"
                }
            },
            "NY": {
                "prior_authorization": {
                    "description": "Specific prior authorization requirements",
                    "requirements": ["Emergency services exemption", "Timely review standards"],
                    "penalty": "medium"
                },
                "external_review": {
                    "description": "External review process requirements",
                    "requirements": ["Independent review organization", "Consumer notification"],
                    "penalty": "high"
                }
            },
            "TX": {
                "medical_privacy": {
                    "description": "Enhanced medical privacy protections",
                    "requirements": ["Additional mental health protections", "Breach notification"],
                    "penalty": "medium"
                },
                "prompt_pay": {
                    "description": "Prompt payment requirements for claims",
                    "requirements": ["30-day payment requirement", "Interest penalties"],
                    "penalty": "high"
                }
            }
        }
    
    def _load_payer_policies(self) -> Dict[str, Dict[str, Any]]:
        """Load payer-specific policies and requirements"""
        
        return {
            "aetna": {
                "prior_authorization": {
                    "threshold": 1000,
                    "required_procedures": ["MRI", "CT", "Surgery", "DME"],
                    "emergency_exemption": True
                },
                "appeal_deadlines": {
                    "initial_appeal": 60,  # days
                    "external_review": 30   # days after denial
                },
                "documentation_requirements": [
                    "Complete medical records",
                    "Provider attestation",
                    "Clinical guidelines reference"
                ]
            },
            "united": {
                "prior_authorization": {
                    "threshold": 750,
                    "required_procedures": ["MRI", "CT", "Surgery", "DME", "PT"],
                    "emergency_exemption": True
                },
                "appeal_deadlines": {
                    "initial_appeal": 45,
                    "external_review": 30
                },
                "documentation_requirements": [
                    "Formal appeal letter",
                    "Complete medical records",
                    "Clinical justification"
                ]
            },
            "bluecross": {
                "prior_authorization": {
                    "threshold": 500,
                    "required_procedures": ["Surgery", "DME", "Specialty drugs"],
                    "emergency_exemption": True
                },
                "appeal_deadlines": {
                    "initial_appeal": 90,
                    "external_review": 60
                },
                "documentation_requirements": [
                    "Provider letter",
                    "Medical necessity documentation",
                    "Treatment history"
                ]
            }
        }
    
    def _load_timely_filing_rules(self) -> Dict[str, int]:
        """Load timely filing requirements by payer/state"""
        
        return {
            "medicare": 365,      # 1 year
            "medicaid": 365,      # 1 year (varies by state)
            "commercial": 180,    # 6 months (typical)
            "workers_comp": 30,   # 30 days (varies by state)
            "default": 180        # Default for unknown payers
        }
    
    def check_claim_compliance(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive compliance check on a claim"""
        
        violations = []
        warnings = []
        recommendations = []
        
        # Check HIPAA compliance
        hipaa_results = self._check_hipaa_compliance(claim_data)
        violations.extend(hipaa_results["violations"])
        warnings.extend(hipaa_results["warnings"])
        
        # Check timely filing
        filing_results = self._check_timely_filing(claim_data)
        if filing_results["violation"]:
            violations.append(filing_results)
        
        # Check state regulations
        state = claim_data.get("state", "")
        if state:
            state_results = self._check_state_compliance(claim_data, state)
            violations.extend(state_results["violations"])
            warnings.extend(state_results["warnings"])
        
        # Check payer-specific policies
        payer = claim_data.get("payer", "").lower()
        if payer in self.payer_policies:
            payer_results = self._check_payer_compliance(claim_data, payer)
            violations.extend(payer_results["violations"])
            warnings.extend(payer_results["warnings"])
        
        # Generate recommendations
        recommendations = self._generate_compliance_recommendations(violations, warnings)
        
        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(violations, warnings)
        
        # Determine overall status
        if violations:
            status = "violation"
        elif warnings:
            status = "warning"
        else:
            status = "compliant"
        
        return {
            "claim_id": claim_data.get("claim_id"),
            "compliance_status": status,
            "compliance_score": compliance_score,
            "violations": violations,
            "warnings": warnings,
            "recommendations": recommendations,
            "checked_at": datetime.now().isoformat(),
            "next_review_date": (datetime.now() + timedelta(days=30)).isoformat()
        }
    
    def _check_hipaa_compliance(self, claim_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Check HIPAA compliance requirements"""
        
        violations = []
        warnings = []
        
        # Check patient consent
        if not claim_data.get("patient_consent"):
            violations.append({
                "type": "HIPAA_VIOLATION",
                "rule": "patient_consent",
                "description": "Patient consent for PHI disclosure not documented",
                "regulation": "45 CFR § 164.508",
                "severity": "high",
                "required_action": "Obtain patient consent before processing appeal"
            })
        
        # Check for appropriate PHI handling
        phi_fields = ["patient_name", "ssn", "dob", "medical_record_number"]
        exposed_phi = []
        for field in phi_fields:
            if claim_data.get(field) and not claim_data.get("phi_protected", False):
                exposed_phi.append(field)
        
        if exposed_phi:
            warnings.append({
                "type": "HIPAA_WARNING",
                "rule": "minimum_necessary",
                "description": f"PHI fields may not be properly protected: {', '.join(exposed_phi)}",
                "regulation": "45 CFR § 164.502(b)",
                "severity": "medium",
                "required_action": "Verify PHI protection measures are in place"
            })
        
        return {"violations": violations, "warnings": warnings}
    
    def _check_timely_filing(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check timely filing compliance"""
        
        service_date_str = claim_data.get("service_date")
        filing_date_str = claim_data.get("filing_date", datetime.now().isoformat())
        payer_type = claim_data.get("payer_type", "commercial").lower()
        
        if not service_date_str:
            return {
                "violation": True,
                "type": "TIMELY_FILING_VIOLATION",
                "description": "Service date not provided - cannot verify timely filing",
                "severity": "high",
                "required_action": "Provide service date for timely filing verification"
            }
        
        try:
            service_date = datetime.fromisoformat(service_date_str.replace('Z', '+00:00'))
            filing_date = datetime.fromisoformat(filing_date_str.replace('Z', '+00:00'))
            
            days_elapsed = (filing_date - service_date).days
            filing_limit = self.timely_filing_rules.get(payer_type, self.timely_filing_rules["default"])
            
            if days_elapsed > filing_limit:
                return {
                    "violation": True,
                    "type": "TIMELY_FILING_VIOLATION",
                    "description": f"Claim filed {days_elapsed} days after service date (limit: {filing_limit} days)",
                    "severity": "high",
                    "days_late": days_elapsed - filing_limit,
                    "required_action": "Document good cause for late filing or consider appeal dismissal"
                }
        
        except (ValueError, TypeError):
            return {
                "violation": True,
                "type": "DATE_FORMAT_ERROR",
                "description": "Invalid date format provided",
                "severity": "medium",
                "required_action": "Correct date format (ISO 8601 required)"
            }
        
        return {"violation": False}
    
    def _check_state_compliance(self, claim_data: Dict[str, Any], state: str) -> Dict[str, List[Dict[str, Any]]]:
        """Check state-specific compliance requirements"""
        
        violations = []
        warnings = []
        
        state_rules = self.state_regulations.get(state.upper(), {})
        
        for rule_name, rule_data in state_rules.items():
            # Check specific state requirements
            if rule_name == "patient_rights" and state.upper() == "CA":
                if not claim_data.get("language_consent"):
                    violations.append({
                        "type": "STATE_REGULATION_VIOLATION",
                        "state": state,
                        "rule": rule_name,
                        "description": "California requires consent in patient's primary language",
                        "severity": rule_data["penalty"],
                        "required_action": "Obtain consent in patient's primary language"
                    })
            
            elif rule_name == "prior_authorization" and state.upper() == "NY":
                amount = claim_data.get("claim_amount", 0)
                if amount > 500 and not claim_data.get("prior_authorization"):
                    warnings.append({
                        "type": "STATE_REGULATION_WARNING",
                        "state": state,
                        "rule": rule_name,
                        "description": "New York may require prior authorization for services >$500",
                        "severity": rule_data["penalty"],
                        "required_action": "Verify prior authorization requirements"
                    })
        
        return {"violations": violations, "warnings": warnings}
    
    def _check_payer_compliance(self, claim_data: Dict[str, Any], payer: str) -> Dict[str, List[Dict[str, Any]]]:
        """Check payer-specific compliance requirements"""
        
        violations = []
        warnings = []
        
        payer_policy = self.payer_policies.get(payer, {})
        
        # Check prior authorization requirements
        prior_auth_policy = payer_policy.get("prior_authorization", {})
        threshold = prior_auth_policy.get("threshold", 0)
        required_procedures = prior_auth_policy.get("required_procedures", [])
        
        claim_amount = claim_data.get("claim_amount", 0)
        procedure_code = claim_data.get("procedure_code", "")
        
        # Check amount threshold
        if claim_amount > threshold and not claim_data.get("prior_authorization"):
            violations.append({
                "type": "PAYER_POLICY_VIOLATION",
                "payer": payer,
                "rule": "prior_authorization_amount",
                "description": f"Prior authorization required for services >${threshold}",
                "severity": "high",
                "required_action": "Obtain prior authorization or document emergency exception"
            })
        
        # Check procedure requirements
        for req_procedure in required_procedures:
            if req_procedure.lower() in procedure_code.lower() and not claim_data.get("prior_authorization"):
                violations.append({
                    "type": "PAYER_POLICY_VIOLATION",
                    "payer": payer,
                    "rule": "prior_authorization_procedure",
                    "description": f"Prior authorization required for {req_procedure} procedures",
                    "severity": "high",
                    "required_action": "Obtain prior authorization for this procedure type"
                })
        
        # Check documentation requirements
        doc_requirements = payer_policy.get("documentation_requirements", [])
        missing_docs = []
        
        for req_doc in doc_requirements:
            doc_key = req_doc.lower().replace(" ", "_")
            if not claim_data.get(doc_key):
                missing_docs.append(req_doc)
        
        if missing_docs:
            warnings.append({
                "type": "PAYER_POLICY_WARNING",
                "payer": payer,
                "rule": "documentation_requirements",
                "description": f"Missing recommended documentation: {', '.join(missing_docs)}",
                "severity": "medium",
                "required_action": "Provide missing documentation to strengthen appeal"
            })
        
        return {"violations": violations, "warnings": warnings}
    
    def _generate_compliance_recommendations(self, violations: List[Dict[str, Any]], 
                                           warnings: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable compliance recommendations"""
        
        recommendations = []
        
        # High-priority recommendations from violations
        for violation in violations:
            if violation.get("severity") == "high":
                recommendations.append(f"URGENT: {violation.get('required_action', 'Address violation')}")
            else:
                recommendations.append(violation.get('required_action', 'Address violation'))
        
        # Medium-priority recommendations from warnings
        for warning in warnings:
            action = warning.get('required_action', 'Review warning')
            if action not in recommendations:  # Avoid duplicates
                recommendations.append(f"RECOMMENDED: {action}")
        
        # General recommendations
        if not violations and not warnings:
            recommendations.append("No compliance issues identified - proceed with confidence")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _calculate_compliance_score(self, violations: List[Dict[str, Any]], 
                                  warnings: List[Dict[str, Any]]) -> float:
        """Calculate overall compliance score (0.0 to 1.0)"""
        
        base_score = 1.0
        
        # Deduct for violations
        for violation in violations:
            severity = violation.get("severity", "medium")
            if severity == "high":
                base_score -= 0.25
            elif severity == "medium":
                base_score -= 0.15
            else:  # low
                base_score -= 0.05
        
        # Deduct for warnings
        for warning in warnings:
            severity = warning.get("severity", "medium")
            if severity == "high":
                base_score -= 0.10
            elif severity == "medium":
                base_score -= 0.05
            else:  # low
                base_score -= 0.02
        
        return round(max(0.0, base_score), 2)
    
    def validate_appeal_submission(self, appeal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate appeal before submission to ensure compliance"""
        
        validation_results = {
            "valid": True,
            "blocking_issues": [],
            "warning_issues": [],
            "recommendations": []
        }
        
        # Check required fields
        required_fields = ["appeal_id", "claim_id", "patient_name", "payer", "appeal_content"]
        missing_fields = [field for field in required_fields if not appeal_data.get(field)]
        
        if missing_fields:
            validation_results["valid"] = False
            validation_results["blocking_issues"].append({
                "type": "MISSING_REQUIRED_FIELDS",
                "description": f"Missing required fields: {', '.join(missing_fields)}",
                "action": "Provide all required information before submission"
            })
        
        # Check appeal deadlines
        denial_date_str = appeal_data.get("denial_date")
        if denial_date_str:
            try:
                denial_date = datetime.fromisoformat(denial_date_str.replace('Z', '+00:00'))
                days_since_denial = (datetime.now() - denial_date).days
                
                payer = appeal_data.get("payer", "").lower()
                payer_policy = self.payer_policies.get(payer, {})
                appeal_deadline = payer_policy.get("appeal_deadlines", {}).get("initial_appeal", 60)
                
                if days_since_denial > appeal_deadline:
                    validation_results["valid"] = False
                    validation_results["blocking_issues"].append({
                        "type": "APPEAL_DEADLINE_EXCEEDED",
                        "description": f"Appeal deadline exceeded by {days_since_denial - appeal_deadline} days",
                        "action": "Document good cause for late appeal or consider case closure"
                    })
                elif days_since_denial > (appeal_deadline * 0.8):  # 80% of deadline
                    validation_results["warning_issues"].append({
                        "type": "APPEAL_DEADLINE_WARNING",
                        "description": f"Appeal due in {appeal_deadline - days_since_denial} days",
                        "action": "Expedite appeal processing"
                    })
            
            except (ValueError, TypeError):
                validation_results["warning_issues"].append({
                    "type": "DATE_FORMAT_WARNING",
                    "description": "Invalid denial date format",
                    "action": "Verify denial date format"
                })
        
        # Generate final recommendations
        if validation_results["valid"]:
            validation_results["recommendations"].append("Appeal meets all compliance requirements")
        else:
            validation_results["recommendations"].append("Address blocking issues before submission")
        
        return validation_results
