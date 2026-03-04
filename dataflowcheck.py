#!/usr/bin/env python3
"""
Insurance API Data Flow Test Script
==================================

This script demonstrates exactly what data is being sent to the insurance APIs
and tests the complete data transformation pipeline.

Understanding the Data Flow:
1. Patient CSV data → Raw claim data
2. Risk Predictor Agent → Risk assessment 
3. Auto-Corrector Agent → Data validation/correction
4. Claim Submitter Agent → API payload formatting
5. Insurance API → Response processing

"""

import asyncio
import httpx
import json
import pandas as pd
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.csv_data_loader import PatientLoader
from config.settings import Settings

class InsuranceAPIDataFlowTester:
    """Test the complete data flow to insurance APIs"""
    
    def __init__(self):
        self.patient_loader = PatientLoader(csv_path="data/patients1.csv")
        self.settings = Settings()
        
    def print_banner(self, title: str):
        """Print formatted banner"""
        print("\n" + "="*80)
        print(f"📊 {title}")
        print("="*80)
    
    def print_section(self, title: str):
        """Print section header"""
        print(f"\n🔍 {title}")
        print("-" * 60)
    
    def get_api_url_for_insurer(self, insurer: str) -> str:
        """Determine which API URL to use based on insurer"""
        insurer_lower = insurer.lower()
        
        # Primary API: BlueCross, Aetna
        if any(name in insurer_lower for name in ['bluecross', 'aetna']):
            return "http://localhost:8081"
        
        # Secondary API: Cigna, United
        elif any(name in insurer_lower for name in ['cigna', 'united']):
            return "http://localhost:8082"
        
        # Default to primary
        return "http://localhost:8081"
    
    def transform_csv_to_claim_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform CSV patient data to claim format (Step 1)"""
        return {
            "claim_id": f"CLM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{patient_data['patient_id']}",
            "patient_id": patient_data["patient_id"],
            "patient_name": patient_data["name"],
            "age": patient_data["age"],
            "gender": patient_data["gender"],
            "date_of_birth": patient_data["dob"],
            "diagnosis": self.get_diagnosis_description(patient_data["diagnosis_code"]),
            "diagnosis_code": patient_data["diagnosis_code"],
            "icd_code": patient_data["diagnosis_code"],
            "procedure_code": patient_data["procedure_code"],
            "cpt_code": patient_data["procedure_code"],
            "claim_amount": patient_data["claim_amount"],
            "insurance_company": patient_data["insurer"],
            "insurer": patient_data["insurer"],
            "service_date": patient_data["service_date"],
            "treatment_date": patient_data["service_date"],
            "provider": patient_data["provider"],
            "provider_npi": self.generate_npi(patient_data["provider"]),
            "medical_history": patient_data["medical_history"],
            "allergies": patient_data["allergies"],
            "medications": patient_data["medications"],
            "prior_auth": patient_data["prior_authorization"],
            "prior_authorization": patient_data["prior_authorization"]
        }
    
    def simulate_risk_assessment(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate risk predictor output (Step 2)"""
        
        # Calculate risk score based on various factors
        risk_score = 0.0
        issues = []
        
        # Check for missing data
        if not claim_data.get("prior_auth"):
            risk_score += 0.3
            issues.append("No prior authorization")
        
        if not claim_data.get("medical_history") or claim_data["medical_history"] == "None":
            risk_score += 0.2
            issues.append("Insufficient medical history")
        
        if claim_data["claim_amount"] > 500:
            risk_score += 0.2
            issues.append("High claim amount")
        
        # Age-based risk
        if claim_data["age"] > 65:
            risk_score += 0.1
            issues.append("Senior patient - additional review needed")
        
        return {
            "risk_score": min(risk_score, 1.0),
            "issues": issues,
            "confidence": 0.85,
            "recommendations": [
                "Verify prior authorization status",
                "Obtain detailed medical documentation",
                "Review claim amount justification"
            ]
        }
    
    def simulate_auto_correction(self, claim_data: Dict[str, Any], risk_result: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate auto-corrector output (Step 3)"""
        
        corrected_data = claim_data.copy()
        corrections_made = []
        
        # Fix missing prior auth
        if not corrected_data.get("prior_auth"):
            corrected_data["prior_auth"] = "Pending verification"
            corrections_made.append("Added prior authorization status")
        
        # Enhance medical history
        if not corrected_data.get("medical_history") or corrected_data["medical_history"] == "None":
            corrected_data["medical_history"] = f"Patient with {corrected_data['diagnosis']} requiring standard care"
            corrections_made.append("Enhanced medical history documentation")
        
        # Add provider NPI if missing
        if not corrected_data.get("provider_npi"):
            corrected_data["provider_npi"] = self.generate_npi(corrected_data["provider"])
            corrections_made.append("Generated provider NPI")
        
        return {
            "corrected_data": corrected_data,
            "corrections_made": corrections_made,
            "correction_confidence": 0.9
        }
    
    def format_api_payload(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format final API payload exactly as sent to insurance APIs (Step 4)"""
        
        return {
            "patient_id": str(claim_data.get("patient_id") or "UNKNOWN"),
            "patient_name": str(claim_data.get("patient_name") or "Unknown Patient"), 
            "diagnosis": str(claim_data.get("diagnosis") or "General examination"),
            "icd_code": str(claim_data.get("icd_code") or "Z00.00"),
            "cpt_code": str(claim_data.get("cpt_code") or claim_data.get("procedure_code") or "99213"),
            "claim_amount": float(claim_data.get("claim_amount") or 0),
            "insurance_company": str(claim_data.get("insurance_company") or "Unknown"),
            "prior_auth": str(claim_data.get("prior_auth") or "Not provided"),
            "medical_history": str(claim_data.get("medical_history") or "No history available"),
            "provider_npi": str(claim_data.get("provider_npi") or "1234567890"),
            "treatment_date": str(claim_data.get("treatment_date") or claim_data.get("service_date") or "2024-01-01")
        }
    
    def get_diagnosis_description(self, icd_code: str) -> str:
        """Convert ICD code to description"""
        icd_mappings = {
            "E11.9": "Type 2 diabetes mellitus without complications",
            "J45.9": "Asthma, unspecified", 
            "I10": "Essential hypertension",
            "G43.9": "Migraine, unspecified",
            "M79.3": "Panniculitis, unspecified",
            "N39.0": "Urinary tract infection, site not specified",
            "K21.9": "Gastro-esophageal reflux disease without esophagitis",
            "J06.9": "Acute upper respiratory infection, unspecified",
            "E78.5": "Hyperlipidemia, unspecified",
            "Z00.00": "Encounter for general adult medical examination without abnormal findings",
            "I25.9": "Chronic ischemic heart disease, unspecified",
            "F32.9": "Major depressive disorder, single episode, unspecified",
            "J44.0": "Chronic obstructive pulmonary disease with acute lower respiratory infection",
            "N80.1": "Endometriosis of ovary",
            "C78.00": "Secondary malignant neoplasm of unspecified lung",
            "Z34.90": "Encounter for supervision of normal pregnancy, unspecified",
            "E10.9": "Type 1 diabetes mellitus without complications",
            "M54.5": "Low back pain",
            "F41.9": "Anxiety disorder, unspecified",
            "E03.9": "Hypothyroidism, unspecified",
            "M25.9": "Joint disorder, unspecified",
            "H40.9": "Unspecified glaucoma",
            "Z51.11": "Encounter for antineoplastic chemotherapy",
            "E66.9": "Obesity, unspecified",
            "N18.6": "End stage renal disease"
        }
        return icd_mappings.get(icd_code, f"Medical condition (ICD: {icd_code})")
    
    def generate_npi(self, provider_name: str) -> str:
        """Generate mock NPI based on provider name"""
        npi_mappings = {
            "Dr. Anderson": "1234567890",
            "Dr. Brown": "1234567891", 
            "Dr. Wilson": "1234567892",
            "Dr. Taylor": "1234567893",
            "Dr. Kim": "1234567894",
            "Dr. Patel": "1234567895",
            "Dr. Chen": "1234567896",
            "Dr. Rodriguez": "1234567897",
            "Dr. Johnson": "1234567898",
            "Dr. Williams": "1234567899"
        }
        return npi_mappings.get(provider_name, "1234567890")
    
    async def test_api_connectivity(self):
        """Test if insurance APIs are running"""
        self.print_section("API Connectivity Test")
        
        apis = [
            ("Primary API (BlueCross/Aetna)", "http://localhost:8081/api/health"),
            ("Secondary API (Cigna/United)", "http://localhost:8082/api/health")
        ]
        
        for name, url in apis:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    response = await client.get(url)
                    if response.status_code == 200:
                        print(f"✅ {name}: ONLINE")
                        print(f"   Response: {response.json()}")
                    else:
                        print(f"")
            except Exception as e:
                print(f"❌ {name}: OFFLINE ({str(e)})")
    
    async def test_complete_data_flow(self, patient_id: str = "PAT001"):
        """Test complete data flow for a specific patient"""
        
        self.print_banner(f"COMPLETE DATA FLOW TEST - Patient {patient_id}")
        
        # Step 1: Load patient data from CSV
        self.print_section("Step 1: Raw Patient Data from CSV")
        patient_data = self.patient_loader.get_patient_by_id(patient_id)
        
        if not patient_data:
            print(f"❌ Patient {patient_id} not found in CSV")
            return
        
        print("📋 Raw CSV Data:")
        for key, value in patient_data.items():
            print(f"   {key}: {repr(value)}")
        
        # Step 2: Transform to claim format
        self.print_section("Step 2: Claim Data Transformation")
        claim_data = self.transform_csv_to_claim_data(patient_data)
        
        print("🔄 Transformed Claim Data:")
        for key, value in claim_data.items():
            print(f"   {key}: {repr(value)}")
        
        # Step 3: Risk assessment
        self.print_section("Step 3: Risk Assessment Simulation")
        risk_result = self.simulate_risk_assessment(claim_data)
        
        print("🎯 Risk Assessment Result:")
        print(f"   Risk Score: {risk_result['risk_score']}")
        print(f"   Issues Found: {len(risk_result['issues'])}")
        for issue in risk_result['issues']:
            print(f"      • {issue}")
        
        # Step 4: Auto-correction
        self.print_section("Step 4: Auto-Correction Simulation")
        correction_result = self.simulate_auto_correction(claim_data, risk_result)
        
        print("🔧 Auto-Correction Results:")
        print(f"   Corrections Made: {len(correction_result['corrections_made'])}")
        for correction in correction_result['corrections_made']:
            print(f"      • {correction}")
        
        corrected_claim = correction_result['corrected_data']
        
        # Step 5: Format API payload
        self.print_section("Step 5: Final API Payload Formatting")
        api_payload = self.format_api_payload(corrected_claim)
        
        print("📦 FINAL API PAYLOAD (Exact data sent to insurance API):")
        print(json.dumps(api_payload, indent=2, ensure_ascii=False))
        
        # Step 6: Determine target API
        insurance_company = api_payload['insurance_company']
        api_url = self.get_api_url_for_insurer(insurance_company)
        
        self.print_section("Step 6: API Routing Decision")
        print(f"Insurance Company: {insurance_company}")
        print(f"Target API URL: {api_url}")
        print(f"API Type: {'Primary (BlueCross/Aetna)' if '8081' in api_url else 'Secondary (Cigna/United)'}")
        
        # Step 7: Actual API submission test
        self.print_section("Step 7: Live API Submission Test")
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                print(f"🚀 Submitting to: {api_url}/submit")
                print(f"📤 Payload size: {len(json.dumps(api_payload))} bytes")
                
                response = await client.post(
                    f"{api_url}/submit",
                    json=api_payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print("✅ API Submission Successful!")
                    print("📥 API Response:")
                    print(json.dumps(result, indent=2))
                    
                    # If pending, check status
                    if result.get("status") == "pending":
                        claim_id = result.get("claim_id")
                        print(f"\n⏳ Checking claim status for: {claim_id}")
                        
                        status_response = await client.get(f"{api_url}/claim-status/{claim_id}")
                        if status_response.status_code == 200:
                            status_result = status_response.json()
                            print("📊 Status Check Response:")
                            print(json.dumps(status_result, indent=2))
                else:
                    print(f"")
                    print(f"Response: {response.text}")
                    
        except Exception as e:
            print(f" {str(e)}")
    
    async def test_multiple_patients(self):
        """Test data flow for multiple patients with different insurers"""
        
        self.print_banner("MULTI-PATIENT DATA FLOW TEST")
        
        # Test different insurance types
        test_patients = ["PAT001", "PAT002", "PAT003", "PAT004"]  # BlueCross, Aetna, Cigna, United
        
        for patient_id in test_patients:
            patient_data = self.patient_loader.get_patient_by_id(patient_id)
            if patient_data:
                print(f"\n🔍 Testing {patient_id} - {patient_data['name']} ({patient_data['insurer']})")
                
                claim_data = self.transform_csv_to_claim_data(patient_data)
                api_payload = self.format_api_payload(claim_data)
                api_url = self.get_api_url_for_insurer(patient_data['insurer'])
                
                print(f"   Insurance: {patient_data['insurer']}")
                print(f"   API Route: {api_url}")
                print(f"   Claim Amount: ${api_payload['claim_amount']}")
                print(f"   Procedure: {api_payload['cpt_code']}")
    
    async def run_comprehensive_test(self):
        """Run all tests"""
        
        print("🏥 INSURANCE API DATA FLOW COMPREHENSIVE TEST")
        print("=" * 80)
        print("This script shows exactly what data flows to the insurance APIs")
        print("and how it's transformed through each step of the process.")
        
        # Test API connectivity first
        await self.test_api_connectivity()
        
        # Test complete flow for one patient in detail
        await self.test_complete_data_flow("PAT001")
        
        # Test multiple patients summary
        await self.test_multiple_patients()
        
        self.print_banner("TEST SUMMARY")
        print("✅ Data Flow Analysis Complete")
        print("📊 Key Insights:")
        print("   • Patient CSV data is transformed through 4 stages")
        print("   • API routing is based on insurance company")
        print("   • Primary API (8081): BlueCross, Aetna")
        print("   • Secondary API (8082): Cigna, United")
        print("   • Final payload contains 11 required fields")
        print("   • All responses are JSON formatted")

async def main():
    """Main function"""
    tester = InsuranceAPIDataFlowTester()
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    print("🚀 Starting Insurance API Data Flow Test...")
    asyncio.run(main())
