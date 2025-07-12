# tools/csv_data_loader.py - Enhanced CSV Data Loader for Patient Management

import pandas as pd
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

class PatientLoader:
    """Enhanced patient data loader with full profile support"""
    
    def __init__(self, csv_path: str = "data/patients.csv"):
        # Convert relative paths to absolute paths
        if not os.path.isabs(csv_path):
            # Get the project root directory (parent of tools/)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.csv_path = os.path.join(project_root, csv_path)
        else:
            self.csv_path = csv_path
        
        self.patients_df = None
        self.load_data()
    
    def load_data(self):
        """Load patient data from CSV"""
        try:
            self.patients_df = pd.read_csv(self.csv_path)
            print(f"[SUCCESS] Loaded {len(self.patients_df)} patients from {self.csv_path}")
        except Exception as e:
            print(f"[ERROR] Error loading patient data: {e}")
            self.patients_df = pd.DataFrame()
    
    def get_all_patients(self) -> List[Dict]:
        """Get all patients as a list of dictionaries"""
        if self.patients_df.empty:
            return []
        
        patients = []
        for _, patient in self.patients_df.iterrows():
            patients.append({
                "patient_id": patient["patient_id"],
                "name": patient["name"],
                "age": patient["age"],
                "gender": patient["gender"],
                "dob": patient["dob"],
                "phone": patient["phone"],
                "email": patient["email"],
                "address": patient["address"],
                "insurer": patient["insurer"],
                "procedure_code": patient["procedure_code"],
                "diagnosis_code": patient["diagnosis_code"],
                "claim_amount": float(patient["claim_amount"]),
                "service_date": patient["service_date"],
                "provider": patient["provider"],
                "medical_history": patient["medical_history"] if pd.notna(patient["medical_history"]) else "",
                "allergies": patient["allergies"] if pd.notna(patient["allergies"]) else "",
                "medications": patient["medications"] if pd.notna(patient["medications"]) else "",
                "prior_authorization": patient["prior_authorization"] == "true" if pd.notna(patient["prior_authorization"]) else False
            })
        
        return patients
    
    def get_patients_with_pending_claims(self) -> List[Dict]:
        """Get all patients (since all start as unclaimed)"""
        return self.get_all_patients()
    
    def get_patient_by_id(self, patient_id: str) -> Optional[Dict]:
        """Get a specific patient by ID"""
        if self.patients_df.empty:
            return None
        
        patient_row = self.patients_df[self.patients_df["patient_id"] == patient_id]
        if patient_row.empty:
            return None
        
        patient = patient_row.iloc[0]
        return {
            "patient_id": patient["patient_id"],
            "name": patient["name"],
            "age": patient["age"],
            "gender": patient["gender"],
            "dob": patient["dob"],
            "phone": patient["phone"],
            "email": patient["email"],
            "address": patient["address"],
            "insurer": patient["insurer"],
            "procedure_code": patient["procedure_code"],
            "diagnosis_code": patient["diagnosis_code"],
            "claim_amount": float(patient["claim_amount"]),
            "service_date": patient["service_date"],
            "provider": patient["provider"],
            "medical_history": patient["medical_history"] if pd.notna(patient["medical_history"]) else "",
            "allergies": patient["allergies"] if pd.notna(patient["allergies"]) else "",
            "medications": patient["medications"] if pd.notna(patient["medications"]) else "",
            "prior_authorization": patient["prior_authorization"] == "true" if pd.notna(patient["prior_authorization"]) else False
        }
    
    def get_patients_by_insurer(self, insurer: str) -> List[Dict]:
        """Get patients by insurance company"""
        if self.patients_df.empty:
            return []
        
        filtered_df = self.patients_df[self.patients_df["insurer"] == insurer]
        patients = []
        
        for _, patient in filtered_df.iterrows():
            patients.append({
                "patient_id": patient["patient_id"],
                "name": patient["name"],
                "age": patient["age"],
                "gender": patient["gender"],
                "dob": patient["dob"],
                "phone": patient["phone"],
                "email": patient["email"],
                "address": patient["address"],
                "insurer": patient["insurer"],
                "procedure_code": patient["procedure_code"],
                "diagnosis_code": patient["diagnosis_code"],
                "claim_amount": float(patient["claim_amount"]),
                "service_date": patient["service_date"],
                "provider": patient["provider"],
                "medical_history": patient["medical_history"] if pd.notna(patient["medical_history"]) else "",
                "allergies": patient["allergies"] if pd.notna(patient["allergies"]) else "",
                "medications": patient["medications"] if pd.notna(patient["medications"]) else "",
                "prior_authorization": patient["prior_authorization"] == "true" if pd.notna(patient["prior_authorization"]) else False
            })
        
        return patients
    
    def generate_claim_id(self, patient_id: str) -> str:
        """Generate a unique claim ID for a patient"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"CLM-{patient_id}-{timestamp}"
    
    def get_random_patient(self) -> Optional[Dict]:
        """Get a random patient for testing"""
        if self.patients_df.empty:
            return None
        
        patient = self.patients_df.sample(1).iloc[0]
        return {
            "patient_id": patient["patient_id"],
            "name": patient["name"],
            "age": patient["age"],
            "gender": patient["gender"],
            "dob": patient["dob"],
            "phone": patient["phone"],
            "email": patient["email"],
            "address": patient["address"],
            "insurer": patient["insurer"],
            "procedure_code": patient["procedure_code"],
            "diagnosis_code": patient["diagnosis_code"],
            "claim_amount": float(patient["claim_amount"]),
            "service_date": patient["service_date"],
            "provider": patient["provider"],
            "medical_history": patient["medical_history"] if pd.notna(patient["medical_history"]) else "",
            "allergies": patient["allergies"] if pd.notna(patient["allergies"]) else "",
            "medications": patient["medications"] if pd.notna(patient["medications"]) else "",
            "prior_authorization": patient["prior_authorization"] == "true" if pd.notna(patient["prior_authorization"]) else False
        }

class DenialLearningLoader:
    """Denial learning pattern loader and manager"""
    
    def __init__(self, csv_path: str = "data/denial_learning.csv"):
        # Convert relative paths to absolute paths
        if not os.path.isabs(csv_path):
            # Get the project root directory (parent of tools/)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.csv_path = os.path.join(project_root, csv_path)
        else:
            self.csv_path = csv_path
        
        self.denial_df = None
        self.load_data()
    
    def load_data(self):
        """Load denial learning data from CSV"""
        try:
            if os.path.exists(self.csv_path):
                self.denial_df = pd.read_csv(self.csv_path)
                print(f"[SUCCESS] Loaded {len(self.denial_df)} denial patterns")
            else:
                # Create empty DataFrame with required columns
                self.denial_df = pd.DataFrame(columns=[
                    'patient_id', 'denial_reason', 'insurer', 'procedure_code',
                    'corrective_action', 'timestamp'
                ])
                print("[INFO] Created empty denial learning database")
        except Exception as e:
            print(f"[ERROR] Error loading denial learning data: {e}")
            self.denial_df = pd.DataFrame(columns=[
                'patient_id', 'denial_reason', 'insurer', 'procedure_code',
                'corrective_action', 'timestamp'
            ])
    
    def get_all_patterns(self) -> List[Dict]:
        """Get all denial learning patterns"""
        if self.denial_df.empty:
            return []
        
        patterns = []
        for _, pattern in self.denial_df.iterrows():
            patterns.append({
                "patient_id": pattern["patient_id"],
                "denial_reason": pattern["denial_reason"],
                "insurer": pattern["insurer"],
                "procedure_code": pattern["procedure_code"],
                "corrective_action": pattern.get("corrective_action", ""),
                "timestamp": pattern.get("timestamp", "")
            })
        
        return patterns
    
    def get_patterns_by_insurer(self, insurer: str) -> List[Dict]:
        """Get denial patterns for a specific insurer"""
        if self.denial_df.empty:
            return []
        
        # Handle both 'insurer' and 'insurance_company' column names
        insurer_column = 'insurer' if 'insurer' in self.denial_df.columns else 'insurance_company'
        filtered_df = self.denial_df[self.denial_df[insurer_column] == insurer]
        patterns = []
        
        for _, pattern in filtered_df.iterrows():
            patterns.append({
                "patient_id": pattern.get("patient_id", pattern.get("claim_id", "")),
                "denial_reason": pattern["denial_reason"],
                "insurer": pattern.get("insurer", pattern.get("insurance_company", "")),
                "procedure_code": pattern.get("procedure_code", ""),
                "corrective_action": pattern.get("corrective_action", pattern.get("solution_applied", "")),
                "timestamp": pattern.get("timestamp", "")
            })
        
        return patterns
    
    def get_patterns_by_procedure(self, procedure_code: str) -> List[Dict]:
        """Get denial patterns for a specific procedure"""
        if self.denial_df.empty:
            return []
        
        filtered_df = self.denial_df[self.denial_df["procedure_code"] == procedure_code]
        patterns = []
        
        for _, pattern in filtered_df.iterrows():
            patterns.append({
                "patient_id": pattern["patient_id"],
                "denial_reason": pattern["denial_reason"],
                "insurer": pattern["insurer"],
                "procedure_code": pattern["procedure_code"],
                "corrective_action": pattern.get("corrective_action", ""),
                "timestamp": pattern.get("timestamp", "")
            })
        
        return patterns
    
    def add_denial_pattern(self, patient_id: str, denial_reason: str, insurer: str, 
                          procedure_code: str, corrective_action: str = ""):
        """Add a new denial pattern to the learning database"""
        new_pattern = {
            "patient_id": patient_id,
            "denial_reason": denial_reason,
            "insurer": insurer,
            "procedure_code": procedure_code,
            "corrective_action": corrective_action,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to DataFrame
        self.denial_df = pd.concat([self.denial_df, pd.DataFrame([new_pattern])], ignore_index=True)
        
        # Save to CSV
        self.save_to_csv()
        
        print(f"[INFO] Added denial pattern: {denial_reason} for {insurer}")
    
    def save_to_csv(self):
        """Save denial patterns to CSV file"""
        try:
            self.denial_df.to_csv(self.csv_path, index=False)
            print(f"[SUCCESS] Saved denial patterns to {self.csv_path}")
        except Exception as e:
            print(f"[ERROR] Error saving denial patterns: {e}")
    
    def get_learning_insights(self) -> Dict:
        """Get insights from the denial learning data"""
        if self.denial_df.empty:
            return {"total_patterns": 0, "top_reasons": [], "top_insurers": []}
        
        total_patterns = len(self.denial_df)
        
        # Top denial reasons
        top_reasons = self.denial_df["denial_reason"].value_counts().head(5).to_dict()
        
        # Top insurers with denials
        top_insurers = self.denial_df["insurer"].value_counts().head(5).to_dict()
        
        return {
            "total_patterns": total_patterns,
            "top_reasons": top_reasons,
            "top_insurers": top_insurers
        }

# Legacy compatibility
class PatientDataLoader(PatientLoader):
    """Legacy class for backward compatibility"""
    pass

# Create default instances for backward compatibility
patient_loader = PatientLoader()
denial_loader = DenialLearningLoader()
