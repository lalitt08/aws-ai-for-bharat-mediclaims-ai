"""
Mock Data Generator for Post-Submission Appeals Dashboard
Generates realistic demo data for showcase purposes
"""

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
import uuid

class MockDataGenerator:
    """Generate realistic mock data for the appeals dashboard demo"""
    
    def __init__(self):
        self.payers = ["aetna", "united", "bluecross", "cigna", "humana", "anthem"]
        self.statuses = ["pending", "active", "approved", "denied"]
        self.priorities = ["high", "medium", "low"]
        
        self.denial_reasons = [
            "Medical necessity not established",
            "Prior authorization required",
            "Out of network provider",
            "Services not covered by plan",
            "Duplicate claim submission",
            "Insufficient documentation provided",
            "Experimental or investigational treatment",
            "Service not rendered as billed",
            "Coding error - incorrect CPT code",
            "Timely filing limit exceeded",
            "Patient not eligible on service date",
            "Coordination of benefits required",
            "Pre-existing condition exclusion",
            "Annual/lifetime benefit maximum exceeded",
            "Non-covered cosmetic procedure"
        ]
        
        self.procedure_codes = [
            {"code": "99213", "description": "Office visit - established patient"},
            {"code": "99214", "description": "Office visit - detailed examination"},
            {"code": "70553", "description": "MRI brain with contrast"},
            {"code": "73721", "description": "MRI knee without contrast"},
            {"code": "45380", "description": "Colonoscopy with biopsy"},
            {"code": "29881", "description": "Arthroscopy knee with meniscectomy"},
            {"code": "64483", "description": "Injection lumbar epidural"},
            {"code": "93306", "description": "Echocardiography complete"},
            {"code": "76700", "description": "Ultrasound abdomen complete"},
            {"code": "36415", "description": "Routine venipuncture"}
        ]
        
        self.patient_first_names = [
            "John", "Mary", "David", "Sarah", "Michael", "Jennifer", "Robert", "Lisa",
            "William", "Karen", "Richard", "Nancy", "Joseph", "Betty", "Thomas", "Helen",
            "Charles", "Sandra", "Christopher", "Donna", "Daniel", "Carol", "Matthew", "Ruth",
            "Anthony", "Sharon", "Mark", "Michelle", "Donald", "Laura", "Steven", "Sarah",
            "Paul", "Kimberly", "Andrew", "Deborah", "Kenneth", "Dorothy", "Kevin", "Lisa"
        ]
        
        self.patient_last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
            "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
            "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
            "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
            "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill"
        ]
        
        self.appeal_strategies = [
            "Medical necessity documentation review",
            "Prior authorization retroactive approval",
            "Network adequacy exception request",
            "Coverage interpretation appeal",
            "Timely filing good cause exception",
            "Clinical guidelines review",
            "Peer-to-peer consultation",
            "External medical review",
            "Administrative correction request",
            "Benefits interpretation clarification"
        ]
    
    def generate_appeals(self, count: int = 100) -> List[Dict[str, Any]]:
        """Generate mock appeals data"""
        
        appeals = []
        base_date = datetime.now()
        
        for i in range(1, count + 1):
            # Generate random dates
            created_date = base_date - timedelta(days=random.randint(1, 180))
            service_date = created_date - timedelta(days=random.randint(7, 60))
            denial_date = created_date - timedelta(days=random.randint(1, 14))
            last_updated = created_date + timedelta(days=random.randint(0, 30))
            
            # Generate patient name
            first_name = random.choice(self.patient_first_names)
            last_name = random.choice(self.patient_last_names)
            patient_name = f"{first_name} {last_name}"
            
            # Generate procedure info
            procedure = random.choice(self.procedure_codes)
            
            # Generate claim amount (influenced by procedure type)
            base_amount = random.randint(200, 5000)
            if "MRI" in procedure["description"]:
                base_amount += random.randint(1000, 3000)
            elif "surgery" in procedure["description"].lower():
                base_amount += random.randint(2000, 8000)
            
            # Generate appeal
            appeal = {
                "appeal_id": f"APL-{str(i).zfill(6)}",
                "claim_id": f"CLM-{str(i * 10 + random.randint(1, 9)).zfill(8)}",
                "patient_name": patient_name,
                "patient_id": f"PAT-{str(random.randint(100000, 999999))}",
                "payer": random.choice(self.payers),
                "status": random.choice(self.statuses),
                "priority": self._calculate_priority(base_amount, denial_date),
                "claim_amount": base_amount,
                "denial_reason": random.choice(self.denial_reasons),
                "procedure_code": procedure["code"],
                "procedure_description": procedure["description"],
                "service_date": service_date.isoformat(),
                "denial_date": denial_date.isoformat(),
                "created_at": created_date.isoformat(),
                "last_updated": last_updated.isoformat(),
                "days_since_denial": (datetime.now() - denial_date).days,
                "success_probability": self._calculate_success_probability(),
                "appeal_strategy": random.choice(self.appeal_strategies),
                "provider_name": f"Dr. {random.choice(self.patient_last_names)}",
                "provider_npi": str(random.randint(1000000000, 9999999999)),
                "diagnosis_code": self._generate_diagnosis_code(),
                "estimated_resolution_days": random.randint(14, 90),
                "compliance_score": round(random.uniform(0.65, 1.0), 2)
            }
            
            appeals.append(appeal)
        
        return appeals
    
    def _calculate_priority(self, amount: int, denial_date: datetime) -> str:
        """Calculate priority based on amount and time sensitivity"""
        
        days_since_denial = (datetime.now() - denial_date).days
        
        # High priority conditions
        if amount > 2000 or days_since_denial > 45:
            return "high"
        elif amount > 1000 or days_since_denial > 20:
            return "medium"
        else:
            return "low"
    
    def _calculate_success_probability(self) -> int:
        """Calculate realistic success probability"""
        
        # Most appeals have 60-85% success rate
        base_probability = random.randint(60, 85)
        
        # Add some variation
        variation = random.randint(-10, 15)
        probability = max(40, min(95, base_probability + variation))
        
        return probability
    
    def _generate_diagnosis_code(self) -> str:
        """Generate realistic ICD-10 diagnosis code"""
        
        common_diagnoses = [
            "M25.561", "Z51.11", "E11.9", "I10", "K21.9",
            "M54.5", "J44.1", "F41.9", "N39.0", "R06.02",
            "M79.3", "K59.00", "R50.9", "M25.511", "G43.909"
        ]
        
        return random.choice(common_diagnoses)
    
    def generate_era_data(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate mock ERA/835 transaction data"""
        
        era_transactions = []
        
        for i in range(1, count + 1):
            transaction_date = datetime.now() - timedelta(days=random.randint(1, 90))
            
            # Generate claims in this ERA
            num_claims = random.randint(1, 10)
            claims = []
            
            for j in range(num_claims):
                claim = {
                    "claim_id": f"CLM-{str(random.randint(10000000, 99999999))}",
                    "patient_name": f"{random.choice(self.patient_first_names)} {random.choice(self.patient_last_names)}",
                    "service_date": (transaction_date - timedelta(days=random.randint(7, 30))).isoformat(),
                    "billed_amount": random.randint(100, 3000),
                    "paid_amount": random.randint(50, 2000),
                    "adjustment_amount": random.randint(0, 500),
                    "status": random.choice(["paid", "denied", "partial"]),
                    "denial_codes": random.sample(["CO-96", "CO-97", "PR-1", "PR-2", "OA-23"], 
                                                 random.randint(0, 2)) if random.random() < 0.3 else []
                }
                claims.append(claim)
            
            era = {
                "era_id": f"ERA-{str(i).zfill(6)}",
                "transaction_date": transaction_date.isoformat(),
                "payer": random.choice(self.payers),
                "total_claims": num_claims,
                "total_billed": sum(claim["billed_amount"] for claim in claims),
                "total_paid": sum(claim["paid_amount"] for claim in claims),
                "total_adjustments": sum(claim["adjustment_amount"] for claim in claims),
                "claims": claims,
                "processed_at": datetime.now().isoformat()
            }
            
            era_transactions.append(era)
        
        return era_transactions
    
    def generate_compliance_data(self) -> Dict[str, Any]:
        """Generate mock compliance metrics"""
        
        return {
            "overall_compliance_score": round(random.uniform(0.85, 0.98), 3),
            "hipaa_compliance": {
                "score": round(random.uniform(0.90, 0.99), 3),
                "violations": random.randint(0, 3),
                "last_audit": (datetime.now() - timedelta(days=random.randint(30, 90))).isoformat()
            },
            "timely_filing": {
                "score": round(random.uniform(0.80, 0.95), 3),
                "violations": random.randint(0, 5),
                "average_filing_days": random.randint(45, 120)
            },
            "documentation": {
                "score": round(random.uniform(0.75, 0.92), 3),
                "missing_docs": random.randint(2, 8),
                "incomplete_forms": random.randint(1, 6)
            },
            "payer_policies": {
                "score": round(random.uniform(0.82, 0.96), 3),
                "policy_violations": random.randint(0, 4),
                "coverage_issues": random.randint(1, 7)
            }
        }
    
    def generate_metrics_data(self) -> Dict[str, Any]:
        """Generate comprehensive metrics for dashboard"""
        
        # Generate 30 days of historical data
        daily_metrics = []
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(30):
            date = base_date + timedelta(days=i)
            daily_metric = {
                "date": date.isoformat()[:10],
                "new_appeals": random.randint(2, 15),
                "processed_appeals": random.randint(1, 12),
                "approved_appeals": random.randint(1, 8),
                "denied_appeals": random.randint(0, 4),
                "success_rate": round(random.uniform(0.65, 0.85), 3),
                "avg_processing_time": round(random.uniform(12.0, 25.0), 1),
                "total_claim_value": random.randint(50000, 200000)
            }
            daily_metrics.append(daily_metric)
        
        return {
            "daily_metrics": daily_metrics,
            "summary": {
                "total_appeals_30d": sum(m["new_appeals"] for m in daily_metrics),
                "total_processed_30d": sum(m["processed_appeals"] for m in daily_metrics),
                "average_success_rate": round(sum(m["success_rate"] for m in daily_metrics) / 30, 3),
                "average_processing_time": round(sum(m["avg_processing_time"] for m in daily_metrics) / 30, 1),
                "total_claim_value_30d": sum(m["total_claim_value"] for m in daily_metrics)
            },
            "denial_patterns": {
                reason: {
                    "count": random.randint(5, 25),
                    "percentage": round(random.uniform(5.0, 20.0), 1),
                    "avg_amount": random.randint(800, 2500),
                    "success_rate": round(random.uniform(0.60, 0.90), 3)
                }
                for reason in random.sample(self.denial_reasons, 8)
            },
            "payer_performance": {
                payer: {
                    "total_appeals": random.randint(10, 50),
                    "success_rate": round(random.uniform(0.60, 0.85), 3),
                    "avg_processing_time": round(random.uniform(15.0, 30.0), 1),
                    "total_value": random.randint(100000, 500000)
                }
                for payer in self.payers
            }
        }
    
    def save_mock_data(self, output_dir: str = "data/mock"):
        """Generate and save all mock data to files"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate appeals data
        appeals = self.generate_appeals(100)
        with open(f"{output_dir}/appeals.json", "w") as f:
            json.dump(appeals, f, indent=2)
        
        # Generate ERA data
        era_data = self.generate_era_data(50)
        with open(f"{output_dir}/era_transactions.json", "w") as f:
            json.dump(era_data, f, indent=2)
        
        # Generate compliance data
        compliance_data = self.generate_compliance_data()
        with open(f"{output_dir}/compliance_metrics.json", "w") as f:
            json.dump(compliance_data, f, indent=2)
        
        # Generate metrics data
        metrics_data = self.generate_metrics_data()
        with open(f"{output_dir}/dashboard_metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"Mock data generated and saved to {output_dir}/")
        print(f"- {len(appeals)} appeals")
        print(f"- {len(era_data)} ERA transactions")
        print(f"- Compliance metrics")
        print(f"- 30 days of dashboard metrics")

if __name__ == "__main__":
    generator = MockDataGenerator()
    generator.save_mock_data()
