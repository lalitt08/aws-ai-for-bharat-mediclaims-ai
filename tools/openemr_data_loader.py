# tools/openemr_data_loader.py - OpenEMR Database Data Loader for Patient Management

import pymysql
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from config.settings import Settings

class OpenEMRPatientLoader:
    """OpenEMR database patient data loader with same interface as CSV loader"""
    
    def __init__(self):
        # Database configuration - matches your 1.py script
        self.db_config = {
            "host": "20.244.89.81",
            "port": 3306,
            "user": "root",
            "password": "root",
            "database": "openemr_db"
        }
        
        self.patients_df = None
        self.last_load_count = 0  # Track last loaded count to reduce spam
        self.load_data()
    
    def _get_db_connection(self):
        """Get database connection"""
        try:
            return pymysql.connect(**self.db_config, cursorclass=pymysql.cursors.DictCursor)
        except Exception as e:
            print(f"[ERROR] Database connection failed: {e}")
            return None
    
    def load_data(self):
        """Load patient data from OpenEMR database"""
        try:
            conn = self._get_db_connection()
            if not conn:
                self.patients_df = pd.DataFrame()
                return
            
            # Test database connection first
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM patient_data")
            result = cursor.fetchone()
            print(f"[DEBUG] Found {result['count']} patients in database")
            
            # SQL query to fetch data in the same format as CSV - simplified and robust
            query = """
            SELECT 
                pd.pubpid as patient_id,
                CONCAT(IFNULL(pd.fname, ''), ' ', IFNULL(pd.lname, '')) as name,
                YEAR(CURDATE()) - YEAR(pd.DOB) as age,
                IFNULL(pd.sex, 'U') as gender,
                DATE_FORMAT(pd.DOB, '%Y-%m-%d') as dob,
                IFNULL(pd.phone_cell, '') as phone,
                IFNULL(pd.email, '') as email,
                CONCAT(IFNULL(pd.street, ''), ', ', IFNULL(pd.city, ''), ', ', IFNULL(pd.state, ''), ' ', IFNULL(pd.postal_code, '')) as address,
                IFNULL(i.provider, 'Unknown') as insurer,
                IFNULL(poc.procedure_code, '99213') as procedure_code,
                IFNULL(po.order_diagnosis, 'Unknown') as diagnosis_code,
                IFNULL(b.fee, 250.00) as claim_amount,
                DATE_FORMAT(IFNULL(po.date_ordered, NOW()), '%Y-%m-%d %H:%i:%s') as service_date,
                IFNULL(pp.name, 'Dr. Unknown') as provider,
                IFNULL(po.clinical_hx, '') as medical_history,
                '' as allergies,
                IFNULL(pr.drug, '') as medications,
                IFNULL(i.accept_assignment, 0) as prior_authorization
            FROM patient_data pd
            LEFT JOIN insurance_data i ON pd.pid = i.pid
            LEFT JOIN procedure_order po ON pd.pid = po.patient_id
            LEFT JOIN procedure_order_code poc ON po.procedure_order_id = poc.procedure_order_id
            LEFT JOIN billing b ON pd.pid = b.pid AND poc.procedure_code = b.code
            LEFT JOIN procedure_providers pp ON po.provider_id = pp.ppid
            LEFT JOIN prescriptions pr ON pd.pid = pr.patient_id
            WHERE pd.pubpid IS NOT NULL AND pd.pubpid != ''
            GROUP BY pd.pid
            ORDER BY pd.pid DESC
            """
            
            # Execute query manually and create DataFrame
            cursor.execute(query)
            results = cursor.fetchall()
            
            if results:
                # Convert to DataFrame manually to avoid SQLAlchemy warnings
                self.patients_df = pd.DataFrame(results)
                print(f"[DEBUG] Retrieved {len(self.patients_df)} rows from database")
                
                # Data type conversions - handle safely
                self.patients_df['age'] = pd.to_numeric(self.patients_df['age'], errors='coerce').fillna(0).astype(int)
                self.patients_df['claim_amount'] = pd.to_numeric(self.patients_df['claim_amount'], errors='coerce').fillna(250.00)
                self.patients_df['prior_authorization'] = self.patients_df['prior_authorization'].astype(bool)
                
                # Fill NaN values
                self.patients_df = self.patients_df.fillna('')
                
                # Only log if count changed to reduce spam
                current_count = len(self.patients_df)
                if current_count != self.last_load_count:
                    print(f"[SUCCESS] Loaded {current_count} patients from OpenEMR database")
                    self.last_load_count = current_count
                else:
                    print(f"[DEBUG] Data unchanged - {current_count} patients")
            else:
                print("[WARNING] No patient data found in OpenEMR database")
                self.patients_df = pd.DataFrame()
                self.last_load_count = 0
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"[ERROR] Error loading patient data from database: {e}")
            self.patients_df = pd.DataFrame()
            self.last_load_count = 0
    
    def reload_data(self):
        """Reload data from database for real-time updates"""
        print("[DEBUG] Checking for OpenEMR database updates...")
        self.load_data()
    
    def get_all_patients(self) -> List[Dict]:
        """Get all patients as a list of dictionaries - Same interface as CSV loader"""
        if self.patients_df is None or self.patients_df.empty:
            return []
        
        patients = []
        for _, patient in self.patients_df.iterrows():
            patients.append({
                "patient_id": str(patient["patient_id"]),
                "name": str(patient["name"]),
                "age": int(patient["age"]) if pd.notna(patient["age"]) else 0,
                "gender": str(patient["gender"]),
                "dob": str(patient["dob"]),
                "phone": str(patient["phone"]),
                "email": str(patient["email"]),
                "address": str(patient["address"]),
                "insurer": str(patient["insurer"]),
                "procedure_code": str(patient["procedure_code"]),
                "diagnosis_code": str(patient["diagnosis_code"]),
                "claim_amount": float(patient["claim_amount"]),
                "service_date": str(patient["service_date"]),
                "provider": str(patient["provider"]),
                "medical_history": str(patient["medical_history"]) if pd.notna(patient["medical_history"]) else "",
                "allergies": str(patient["allergies"]) if pd.notna(patient["allergies"]) else "",
                "medications": str(patient["medications"]) if pd.notna(patient["medications"]) else "",
                "prior_authorization": bool(patient["prior_authorization"]) if pd.notna(patient["prior_authorization"]) else False
            })
        
        return patients
    
    def get_patients_with_pending_claims(self) -> List[Dict]:
        """Get all patients (since all start as unclaimed) - Same interface as CSV loader"""
        return self.get_all_patients()
    
    def get_patient_by_id(self, patient_id: str) -> Optional[Dict]:
        """Get a specific patient by ID - Same interface as CSV loader"""
        if self.patients_df is None or self.patients_df.empty:
            return None
        
        patient_row = self.patients_df[self.patients_df["patient_id"] == patient_id]
        if patient_row.empty:
            return None
        
        patient = patient_row.iloc[0]
        return {
            "patient_id": str(patient["patient_id"]),
            "name": str(patient["name"]),
            "age": int(patient["age"]) if pd.notna(patient["age"]) else 0,
            "gender": str(patient["gender"]),
            "dob": str(patient["dob"]),
            "phone": str(patient["phone"]),
            "email": str(patient["email"]),
            "address": str(patient["address"]),
            "insurer": str(patient["insurer"]),
            "procedure_code": str(patient["procedure_code"]),
            "diagnosis_code": str(patient["diagnosis_code"]),
            "claim_amount": float(patient["claim_amount"]),
            "service_date": str(patient["service_date"]),
            "provider": str(patient["provider"]),
            "medical_history": str(patient["medical_history"]) if pd.notna(patient["medical_history"]) else "",
            "allergies": str(patient["allergies"]) if pd.notna(patient["allergies"]) else "",
            "medications": str(patient["medications"]) if pd.notna(patient["medications"]) else "",
            "prior_authorization": bool(patient["prior_authorization"]) if pd.notna(patient["prior_authorization"]) else False
        }
    
    def get_patients_by_insurer(self, insurer: str) -> List[Dict]:
        """Get patients by insurance company - Same interface as CSV loader"""
        if self.patients_df is None or self.patients_df.empty:
            return []
        
        filtered_df = self.patients_df[self.patients_df["insurer"] == insurer]
        patients = []
        
        for _, patient in filtered_df.iterrows():
            patients.append({
                "patient_id": str(patient["patient_id"]),
                "name": str(patient["name"]),
                "age": int(patient["age"]) if pd.notna(patient["age"]) else 0,
                "gender": str(patient["gender"]),
                "dob": str(patient["dob"]),
                "phone": str(patient["phone"]),
                "email": str(patient["email"]),
                "address": str(patient["address"]),
                "insurer": str(patient["insurer"]),
                "procedure_code": str(patient["procedure_code"]),
                "diagnosis_code": str(patient["diagnosis_code"]),
                "claim_amount": float(patient["claim_amount"]),
                "service_date": str(patient["service_date"]),
                "provider": str(patient["provider"]),
                "medical_history": str(patient["medical_history"]) if pd.notna(patient["medical_history"]) else "",
                "allergies": str(patient["allergies"]) if pd.notna(patient["allergies"]) else "",
                "medications": str(patient["medications"]) if pd.notna(patient["medications"]) else "",
                "prior_authorization": bool(patient["prior_authorization"]) if pd.notna(patient["prior_authorization"]) else False
            })
        
        return patients
    
    def generate_claim_id(self, patient_id: str) -> str:
        """Generate a unique claim ID for a patient - Same interface as CSV loader"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"CLM-{patient_id}-{timestamp}"
    
    def get_random_patient(self) -> Optional[Dict]:
        """Get a random patient for testing - Same interface as CSV loader"""
        if self.patients_df is None or self.patients_df.empty:
            return None
        
        patient = self.patients_df.sample(1).iloc[0]
        return {
            "patient_id": str(patient["patient_id"]),
            "name": str(patient["name"]),
            "age": int(patient["age"]) if pd.notna(patient["age"]) else 0,
            "gender": str(patient["gender"]),
            "dob": str(patient["dob"]),
            "phone": str(patient["phone"]),
            "email": str(patient["email"]),
            "address": str(patient["address"]),
            "insurer": str(patient["insurer"]),
            "procedure_code": str(patient["procedure_code"]),
            "diagnosis_code": str(patient["diagnosis_code"]),
            "claim_amount": float(patient["claim_amount"]),
            "service_date": str(patient["service_date"]),
            "provider": str(patient["provider"]),
            "medical_history": str(patient["medical_history"]) if pd.notna(patient["medical_history"]) else "",
            "allergies": str(patient["allergies"]) if pd.notna(patient["allergies"]) else "",
            "medications": str(patient["medications"]) if pd.notna(patient["medications"]) else "",
            "prior_authorization": bool(patient["prior_authorization"]) if pd.notna(patient["prior_authorization"]) else False
        }

    def reload_data(self):
        """Reload data from database - for refresh functionality"""
        print("[INFO] 🔄 Refreshing data from OpenEMR database...")
        self.load_data()

    # Additional methods to maintain compatibility
    def get_patients_df(self):
        """Get the patients DataFrame for compatibility"""
        return self.patients_df
    
    def get_total_patients(self) -> int:
        """Get total number of patients"""
        return len(self.patients_df) if self.patients_df is not None else 0


# Compatibility class to maintain same interface as existing DenialLearningLoader
class OpenEMRDenialLearningLoader:
    """OpenEMR denial learning data loader (placeholder for now)"""
    
    def __init__(self):
        # For now, use the existing CSV denial learning data
        # Can be enhanced later to use database
        try:
            from tools.csv_data_loader import DenialLearningLoader
            self.csv_loader = DenialLearningLoader()
        except:
            self.csv_loader = None
    
    def get_denial_patterns(self):
        """Get denial patterns"""
        if self.csv_loader:
            return self.csv_loader.get_denial_patterns()
        return []
    
    def get_learned_solutions(self, insurer=None):
        """Get learned solutions"""
        if self.csv_loader:
            return self.csv_loader.get_learned_solutions(insurer)
        return []
    
    def reload_data(self):
        """Reload denial learning data"""
        if self.csv_loader:
            self.csv_loader.reload_data()


# Create instances for backward compatibility
def create_openemr_patient_loader():
    """Factory function to create OpenEMR patient loader"""
    return OpenEMRPatientLoader()

def create_openemr_denial_loader():
    """Factory function to create OpenEMR denial loader"""
    return OpenEMRDenialLearningLoader()


# Global instances for backward compatibility
openemr_patient_loader = OpenEMRPatientLoader()
openemr_denial_loader = OpenEMRDenialLearningLoader()
