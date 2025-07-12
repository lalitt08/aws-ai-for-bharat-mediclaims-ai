# tools/medical_knowledge_base.py - Advanced Medical Knowledge Base for MCP

import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

class MedicalKnowledgeBase:
    """
    Advanced medical knowledge base providing ICD codes, CPT codes,
    drug interactions, and clinical guidelines for AI agents
    """
    
    def __init__(self):
        self.icd_codes = self._load_icd_codes()
        self.cpt_codes = self._load_cpt_codes()
        self.drug_interactions = self._load_drug_interactions()
        self.clinical_guidelines = self._load_clinical_guidelines()
        # Add conditions attribute for compatibility
        self.conditions = list(self.icd_codes.keys())
        
    def _load_icd_codes(self) -> Dict[str, Any]:
        """Load ICD-10 code database"""
        return {
            "E11.9": {
                "description": "Type 2 diabetes mellitus without complications",
                "category": "Endocrine, nutritional and metabolic diseases",
                "severity": "moderate",
                "complications": [],
                "related_codes": ["E11.0", "E11.1", "E11.2"],
                "typical_treatments": ["Metformin", "Lifestyle modification", "Blood glucose monitoring"]
            },
            "J45.9": {
                "description": "Asthma, unspecified",
                "category": "Diseases of the respiratory system",
                "severity": "mild to moderate",
                "complications": ["Acute exacerbation", "Status asthmaticus"],
                "related_codes": ["J45.0", "J45.1", "J45.2"],
                "typical_treatments": ["Bronchodilators", "Corticosteroids", "Allergy management"]
            },
            "I10": {
                "description": "Essential (primary) hypertension",
                "category": "Diseases of the circulatory system",
                "severity": "moderate",
                "complications": ["Heart disease", "Stroke", "Kidney disease"],
                "related_codes": ["I11", "I12", "I13"],
                "typical_treatments": ["ACE inhibitors", "Diuretics", "Lifestyle changes"]
            },
            "G43.9": {
                "description": "Migraine, unspecified",
                "category": "Diseases of the nervous system",
                "severity": "mild to severe",
                "complications": ["Status migrainosus", "Medication overuse headache"],
                "related_codes": ["G43.0", "G43.1", "G43.2"],
                "typical_treatments": ["Triptans", "Preventive medications", "Lifestyle modifications"]
            },
            "M79.3": {
                "description": "Panniculitis, unspecified",
                "category": "Diseases of the musculoskeletal system",
                "severity": "mild to moderate",
                "complications": ["Chronic pain", "Mobility issues"],
                "related_codes": ["M79.0", "M79.1", "M79.2"],
                "typical_treatments": ["NSAIDs", "Physical therapy", "Corticosteroids"]
            }
        }
    
    def _load_cpt_codes(self) -> Dict[str, Any]:
        """Load CPT procedure code database"""
        return {
            "99213": {
                "description": "Office or other outpatient visit for the evaluation and management of an established patient",
                "category": "Evaluation and Management",
                "complexity": "low to moderate",
                "typical_duration": "15 minutes",
                "prerequisites": ["Established patient relationship"],
                "documentation_requirements": [
                    "Problem focused history",
                    "Problem focused examination",
                    "Medical decision making of low complexity"
                ],
                "average_reimbursement": {
                    "BlueCross": 125.00,
                    "Aetna": 130.00,
                    "Cigna": 128.00,
                    "United": 132.00
                }
            },
            "99214": {
                "description": "Office or other outpatient visit for the evaluation and management of an established patient (moderate complexity)",
                "category": "Evaluation and Management",
                "complexity": "moderate",
                "typical_duration": "25 minutes",
                "prerequisites": ["Established patient relationship"],
                "documentation_requirements": [
                    "Detailed history",
                    "Detailed examination",
                    "Medical decision making of moderate complexity"
                ],
                "average_reimbursement": {
                    "BlueCross": 185.00,
                    "Aetna": 190.00,
                    "Cigna": 188.00,
                    "United": 192.00
                }
            },
            "99215": {
                "description": "Office or other outpatient visit for the evaluation and management of an established patient (high complexity)",
                "category": "Evaluation and Management",
                "complexity": "high",
                "typical_duration": "40 minutes",
                "prerequisites": ["Established patient relationship"],
                "documentation_requirements": [
                    "Comprehensive history",
                    "Comprehensive examination",
                    "Medical decision making of high complexity"
                ],
                "average_reimbursement": {
                    "BlueCross": 275.00,
                    "Aetna": 280.00,
                    "Cigna": 278.00,
                    "United": 282.00
                }
            },
            "94640": {
                "description": "Pressurized or nonpressurized inhalation treatment for acute airway obstruction",
                "category": "Pulmonary Function",
                "complexity": "moderate",
                "typical_duration": "10-15 minutes",
                "prerequisites": ["Respiratory distress", "Physician supervision"],
                "documentation_requirements": [
                    "Indication for treatment",
                    "Medications administered",
                    "Patient response"
                ],
                "average_reimbursement": {
                    "BlueCross": 45.00,
                    "Aetna": 48.00,
                    "Cigna": 46.00,
                    "United": 47.00
                }
            }
        }
    
    def _load_drug_interactions(self) -> Dict[str, Any]:
        """Load drug interaction database"""
        return {
            "Metformin": {
                "contraindications": ["Severe kidney disease", "Severe liver disease"],
                "interactions": [
                    {
                        "drug": "Contrast dye",
                        "severity": "major",
                        "effect": "Increased risk of lactic acidosis"
                    },
                    {
                        "drug": "Alcohol",
                        "severity": "moderate",
                        "effect": "Increased risk of lactic acidosis"
                    }
                ],
                "monitoring": ["Kidney function", "Vitamin B12 levels"]
            },
            "Lisinopril": {
                "contraindications": ["Angioedema", "Pregnancy", "Bilateral renal artery stenosis"],
                "interactions": [
                    {
                        "drug": "Potassium supplements",
                        "severity": "moderate",
                        "effect": "Hyperkalemia"
                    },
                    {
                        "drug": "NSAIDs",
                        "severity": "moderate",
                        "effect": "Reduced antihypertensive effect"
                    }
                ],
                "monitoring": ["Blood pressure", "Kidney function", "Potassium levels"]
            },
            "Albuterol": {
                "contraindications": ["Hypersensitivity to albuterol"],
                "interactions": [
                    {
                        "drug": "Beta-blockers",
                        "severity": "moderate",
                        "effect": "Reduced bronchodilator effect"
                    },
                    {
                        "drug": "Digoxin",
                        "severity": "minor",
                        "effect": "Decreased digoxin levels"
                    }
                ],
                "monitoring": ["Heart rate", "Blood pressure", "Potassium levels"]
            }
        }
    
    def _load_clinical_guidelines(self) -> Dict[str, Any]:
        """Load clinical guidelines database"""
        return {
            "diabetes_management": {
                "guideline": "American Diabetes Association 2024",
                "recommendations": [
                    "HbA1c target <7% for most adults",
                    "Blood pressure target <140/90 mmHg",
                    "Statin therapy for cardiovascular risk reduction",
                    "Annual eye and foot examinations"
                ],
                "monitoring": ["HbA1c every 3-6 months", "Blood pressure at each visit"],
                "referrals": ["Endocrinology if HbA1c >9%", "Ophthalmology annually"]
            },
            "hypertension_management": {
                "guideline": "American Heart Association 2023",
                "recommendations": [
                    "Blood pressure target <130/80 mmHg",
                    "Lifestyle modifications first-line",
                    "ACE inhibitors or ARBs for most patients",
                    "Thiazide diuretics as add-on therapy"
                ],
                "monitoring": ["Blood pressure at each visit", "Annual lipid panel"],
                "referrals": ["Cardiology if resistant hypertension"]
            },
            "asthma_management": {
                "guideline": "Global Initiative for Asthma 2024",
                "recommendations": [
                    "Step-wise approach to treatment",
                    "Inhaled corticosteroids as first-line",
                    "Bronchodilators for symptom relief",
                    "Asthma action plan for all patients"
                ],
                "monitoring": ["Peak flow measurements", "Symptom control assessment"],
                "referrals": ["Pulmonology if severe or uncontrolled"]
            }
        }
    
    async def get_icd_details(self, icd_code: str) -> Dict[str, Any]:
        """Get detailed ICD code information"""
        await asyncio.sleep(0.1)  # Simulate API call
        return self.icd_codes.get(icd_code, {
            "description": f"ICD code {icd_code}",
            "category": "Unknown",
            "severity": "unknown",
            "error": "Code not found in knowledge base"
        })
    
    async def get_cpt_details(self, cpt_code: str) -> Dict[str, Any]:
        """Get detailed CPT code information"""
        await asyncio.sleep(0.1)  # Simulate API call
        return self.cpt_codes.get(cpt_code, {
            "description": f"CPT code {cpt_code}",
            "category": "Unknown",
            "complexity": "unknown",
            "error": "Code not found in knowledge base"
        })
    
    async def get_cpt_description(self, cpt_code: str) -> str:
        """Get CPT code description"""
        cpt_data = await self.get_cpt_details(cpt_code)
        return cpt_data.get("description", f"CPT code {cpt_code}")
    
    async def check_drug_interactions(self, medications: str) -> List[Dict[str, Any]]:
        """Check for drug interactions"""
        await asyncio.sleep(0.1)  # Simulate API call
        
        interactions = []
        if not medications:
            return interactions
        
        # Parse medications (simplified)
        med_list = [med.strip() for med in medications.split(',')]
        
        for med in med_list:
            for drug_name, drug_data in self.drug_interactions.items():
                if drug_name.lower() in med.lower():
                    interactions.append({
                        "medication": drug_name,
                        "contraindications": drug_data.get("contraindications", []),
                        "interactions": drug_data.get("interactions", []),
                        "monitoring": drug_data.get("monitoring", [])
                    })
        
        return interactions
    
    async def get_clinical_guidelines(self, condition: str) -> Dict[str, Any]:
        """Get clinical guidelines for a condition"""
        await asyncio.sleep(0.1)  # Simulate API call
        
        # Map condition to guideline
        condition_map = {
            "diabetes": "diabetes_management",
            "hypertension": "hypertension_management",
            "asthma": "asthma_management"
        }
        
        guideline_key = condition_map.get(condition.lower())
        if guideline_key:
            return self.clinical_guidelines.get(guideline_key, {})
        
        return {
            "error": f"No clinical guidelines found for {condition}"
        }
    
    async def validate_diagnosis(self, diagnosis_code: str) -> Dict[str, Any]:
        """Validate diagnosis code"""
        await asyncio.sleep(0.1)  # Simulate validation
        
        icd_data = await self.get_icd_details(diagnosis_code)
        
        if "error" in icd_data:
            return {
                "valid": False,
                "diagnosis_code": diagnosis_code,
                "error": "Invalid or unknown diagnosis code"
            }
        
        return {
            "valid": True,
            "diagnosis_code": diagnosis_code,
            "description": icd_data.get("description", ""),
            "category": icd_data.get("category", ""),
            "severity": icd_data.get("severity", "")
        }
    
    async def get_all_knowledge(self) -> Dict[str, Any]:
        """Get all knowledge base data"""
        return {
            "icd_codes": self.icd_codes,
            "cpt_codes": self.cpt_codes,
            "drug_interactions": self.drug_interactions,
            "clinical_guidelines": self.clinical_guidelines,
            "last_updated": datetime.now().isoformat()
        }
