"""
Professional Healthcare Dashboard API Server
Integrated with Full Agentic Claims Processing System
"""

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import json
import asyncio
from datetime import datetime, timedelta
import os
import sys
import logging
from typing import Dict, List, Any

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agentic system components
try:
    from tools.csv_data_loader import PatientLoader, DenialLearningLoader
    from graph.claim_flow import ClaimFlow
    from config.settings import Settings
    AGENTIC_AVAILABLE = True
    HAS_SETTINGS = True
    HAS_CLAIM_FLOW = True
    logger.info("[SUCCESS] Agentic system components imported successfully")
except ImportError as e:
    logger.warning(f"[WARNING] Some agentic components not available: {e}")
    # Try individual imports
    HAS_SETTINGS = False
    HAS_CLAIM_FLOW = False
    try:
        from tools.csv_data_loader import PatientLoader, DenialLearningLoader
        AGENTIC_AVAILABLE = True
        logger.info("[SUCCESS] Data loaders imported successfully")
    except ImportError as e2:
        logger.warning(f"[WARNING] Data loaders not available: {e2}")
        AGENTIC_AVAILABLE = False
    
    # Try Settings separately
    try:
        from config.settings import Settings
        HAS_SETTINGS = True
        logger.info("[SUCCESS] Settings imported successfully")
    except ImportError as e3:
        logger.warning(f"[WARNING] Settings not available: {e3}")
        HAS_SETTINGS = False
    
    # Try ClaimFlow separately
    try:
        from graph.claim_flow import ClaimFlow
        HAS_CLAIM_FLOW = True
        logger.info("[SUCCESS] ClaimFlow imported successfully")
    except ImportError as e4:
        logger.warning(f"[WARNING] ClaimFlow not available: {e4}")
        HAS_CLAIM_FLOW = False

app = Flask(__name__)
CORS(app)

# Add request logging
import logging
logging.basicConfig(level=logging.DEBUG)

@app.before_request
def log_request_info():
    logger.info(f"🌐 Request: {request.method} {request.url}")
    logger.info(f"🌐 Headers: {dict(request.headers)}")

@app.after_request
def log_response_info(response):
    logger.info(f"🌐 Response: {response.status_code}")
    return response

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
WEB_DIR = os.path.dirname(__file__)

# Ensure DATA_DIR exists and is correct
if not os.path.exists(DATA_DIR):
    # Try alternative paths
    alt_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    if os.path.exists(alt_data_dir):
        DATA_DIR = alt_data_dir
    else:
        # Create data directory if it doesn't exist
        os.makedirs(DATA_DIR, exist_ok=True)

# Resolve to absolute path
DATA_DIR = os.path.abspath(DATA_DIR)

logger.info(f"📁 Data directory: {DATA_DIR}")
logger.info(f"📁 Web directory: {WEB_DIR}")

# Check if key files exist
patients_file = os.path.join(DATA_DIR, 'patients.csv')
denials_file = os.path.join(DATA_DIR, 'denial_learning.csv')

logger.info(f"📄 Patients file: {patients_file} (exists: {os.path.exists(patients_file)})")
logger.info(f"📄 Denials file: {denials_file} (exists: {os.path.exists(denials_file)})")

class HealthcareDashboardAPI:
    """Professional Healthcare Dashboard API integrated with Agentic System"""
    
    def __init__(self):
        self.patients_df = None
        self.denials_df = None
        
        # Initialize agentic system components if available
        if AGENTIC_AVAILABLE:
            try:
                # Initialize Settings if available
                if HAS_SETTINGS:
                    self.settings = Settings()
                    logger.info("[SUCCESS] Settings initialized")
                else:
                    self.settings = None
                    logger.info("[WARNING] Settings not available, using defaults")
                
                # Pass the correct data directory paths to the loaders
                patients_file = os.path.join(DATA_DIR, 'patients.csv')
                denials_file = os.path.join(DATA_DIR, 'denial_learning.csv')
                
                self.patient_loader = PatientLoader(csv_path=patients_file)
                self.denial_loader = DenialLearningLoader(csv_path=denials_file)
                
                # Initialize ClaimFlow if available
                if HAS_CLAIM_FLOW:
                    self.claim_flow = ClaimFlow()
                    logger.info("[SUCCESS] ClaimFlow initialized")
                else:
                    self.claim_flow = None
                    logger.info("[WARNING] ClaimFlow not available")
                
                logger.info("[SUCCESS] Agentic system components initialized")
                self.agentic_available = True
            except Exception as e:
                logger.error(f"[ERROR] Error initializing agentic system: {e}")
                self.agentic_available = False
        else:
            self.agentic_available = False
        
        self.load_data()
    
    def load_data(self):
        """Load patient and denial data from CSV files"""
        try:
            if self.agentic_available and hasattr(self, 'patient_loader'):
                # Load data using agentic system
                # Note: PatientLoader loads CSV internally, so we use the loaded DataFrame
                self.patients_df = self.patient_loader.patients_df
                self.denials_df = self.denial_loader.denial_df  # Note: DenialLearningLoader uses 'denial_df'
                logger.info("[SUCCESS] Data loaded using agentic system")
            else:
                # Fallback to direct CSV loading
                self.load_data_fallback()
                
        except Exception as e:
            logger.error(f"[ERROR] Error loading data: {e}")
            self.load_data_fallback()
    
    def reload_data(self):
        """Reload data from CSV files - for real-time updates"""
        logger.info("🔄 Reloading data from CSV files...")
        try:
            if self.agentic_available and hasattr(self, 'patient_loader'):
                # Reload the CSV data
                self.patient_loader = PatientLoader(csv_path=os.path.join(DATA_DIR, 'patients.csv'))
                self.denial_loader = DenialLearningLoader(csv_path=os.path.join(DATA_DIR, 'denial_learning.csv'))
                
                self.patients_df = self.patient_loader.patients_df
                self.denials_df = self.denial_loader.denial_df
                logger.info(f"[SUCCESS] Reloaded {len(self.patients_df)} patients from CSV")
            else:
                self.load_data_fallback()
        except Exception as e:
            logger.error(f"[ERROR] Error reloading data: {e}")
            self.load_data_fallback()
            logger.error(f"[ERROR] Error loading data: {str(e)}")
            self.load_data_fallback()
    
    def load_data_fallback(self):
        """Fallback method to load data directly from CSV files"""
        try:
            # Load patient data
            patients_file = os.path.join(DATA_DIR, 'patients.csv')
            if os.path.exists(patients_file):
                self.patients_df = pd.read_csv(patients_file)
                logger.info(f"[SUCCESS] Loaded {len(self.patients_df)} patient records")
            else:
                logger.warning(f"[WARNING] Patient data file not found: {patients_file}")
                self.generate_sample_data()
            
            # Load denial data
            denials_file = os.path.join(DATA_DIR, 'denial_learning.csv')
            if os.path.exists(denials_file):
                self.denials_df = pd.read_csv(denials_file)
                logger.info(f"[SUCCESS] Loaded {len(self.denials_df)} denial records")
            else:
                logger.warning(f"[WARNING] Denial data file not found: {denials_file}")
                self.generate_sample_denial_data()
                
        except Exception as e:
            logger.error(f"[ERROR] Error in fallback data loading: {str(e)}")
            self.generate_sample_data()
    
    def generate_sample_data(self):
        """Generate sample patient data for demonstration"""
        logger.info("📝 Generating sample patient data...")
        
        sample_patients = [
            {
                'patient_id': 'PAT001',
                'name': 'John Smith',
                'age': 45,
                'gender': 'Male',
                'insurer': 'BlueCross',
                'procedure_code': '99213',
                'diagnosis_code': 'J06.9',
                'claim_amount': 250.00,
                'service_date': '2025-07-10',
                'status': 'Approved'
            },
            {
                'patient_id': 'PAT002',
                'name': 'Sarah Johnson',
                'age': 32,
                'gender': 'Female',
                'insurer': 'Aetna',
                'procedure_code': '94640',
                'diagnosis_code': 'J45.9',
                'claim_amount': 180.00,
                'service_date': '2025-07-09',
                'status': 'Pending'
            },
            {
                'patient_id': 'PAT003',
                'name': 'Michael Davis',
                'age': 58,
                'gender': 'Male',
                'insurer': 'Cigna',
                'procedure_code': '99214',
                'diagnosis_code': 'I10',
                'claim_amount': 320.00,
                'service_date': '2025-07-08',
                'status': 'Approved'
            },
            {
                'patient_id': 'PAT004',
                'name': 'Emma Wilson',
                'age': 28,
                'gender': 'Female',
                'insurer': 'United',
                'procedure_code': '99212',
                'diagnosis_code': 'R50.9',
                'claim_amount': 150.00,
                'service_date': '2025-07-07',
                'status': 'Pending'
            },
            {
                'patient_id': 'PAT005',
                'name': 'Robert Brown',
                'age': 65,
                'gender': 'Male',
                'insurer': 'BlueCross',
                'procedure_code': '99215',
                'diagnosis_code': 'E11.9',
                'claim_amount': 420.00,
                'service_date': '2025-07-06',
                'status': 'Approved'
            }
        ]
        
        # Add more sample patients to reach 25 total
        for i in range(6, 26):
            sample_patients.append({
                'patient_id': f'PAT{i:03d}',
                'name': f'Patient {i}',
                'age': 20 + (i % 50),
                'gender': 'Male' if i % 2 == 0 else 'Female',
                'insurer': ['BlueCross', 'Aetna', 'Cigna', 'United', 'Humana'][i % 5],
                'procedure_code': ['99213', '99214', '99215', '94640', '99212'][i % 5],
                'diagnosis_code': ['J06.9', 'J45.9', 'I10', 'R50.9', 'E11.9'][i % 5],
                'claim_amount': 100 + (i * 15),
                'service_date': f'2025-07-{min(12, i % 12 + 1):02d}',
                'status': ['Approved', 'Pending', 'Approved', 'Pending'][i % 4]
            })
        
        self.patients_df = pd.DataFrame(sample_patients)
        logger.info(f"[SUCCESS] Generated {len(self.patients_df)} sample patient records")
    
    def generate_sample_denial_data(self):
        """Generate sample denial data for demonstration"""
        logger.info("📝 Generating sample denial data...")
        
        sample_denials = [
            {
                'denial_id': 'DEN001',
                'patient_id': 'PAT001',
                'denial_category': 'Prior Authorization',
                'denial_reason': 'Missing prior authorization',
                'timestamp': '2025-07-01',
                'resolved': True
            },
            {
                'denial_id': 'DEN002',
                'patient_id': 'PAT002',
                'denial_category': 'Medical Necessity',
                'denial_reason': 'Procedure not medically necessary',
                'timestamp': '2025-07-02',
                'resolved': False
            },
            {
                'denial_id': 'DEN003',
                'patient_id': 'PAT003',
                'denial_category': 'Documentation',
                'denial_reason': 'Insufficient documentation',
                'timestamp': '2025-07-03',
                'resolved': True
            }
        ]
        
        # Add more sample denials to reach 10 total
        for i in range(4, 11):
            sample_denials.append({
                'denial_id': f'DEN{i:03d}',
                'patient_id': f'PAT{i:03d}',
                'denial_category': ['Prior Authorization', 'Medical Necessity', 'Documentation', 'Coding Error'][i % 4],
                'denial_reason': f'Denial reason {i}',
                'timestamp': f'2025-07-{min(12, i):02d}',
                'resolved': i % 2 == 0
            })
        
        self.denials_df = pd.DataFrame(sample_denials)
        logger.info(f"[SUCCESS] Generated {len(self.denials_df)} sample denial records")
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Calculate simple dashboard metrics for claims processing"""
        # Return zero metrics at startup - will be updated by actions
        return {
            'recovered_amount': 0,
            'claims_applied': 0,
            'active_claims': 0
        }
    
    def get_patients_data(self) -> List[Dict[str, Any]]:
        """Get patient data for dashboard table"""
        try:
            logger.info("🔍 Starting get_patients_data")
            if self.patients_df is None or self.patients_df.empty:
                logger.warning("⚠️ No patient data available")
                return []
            
            logger.info(f"📊 Processing {len(self.patients_df)} patient records")
            
            # Replace NaN values with None (which converts to null in JSON)
            logger.info("🔧 Cleaning NaN values...")
            # Use replace method instead of fillna for better compatibility
            df_cleaned = self.patients_df.replace({pd.NA: None, pd.NaT: None})
            # Also handle numpy NaN values
            import numpy as np
            df_cleaned = df_cleaned.replace({np.nan: None})
            
            # Convert DataFrame to list of dictionaries
            logger.info("🔄 Converting to dictionary records...")
            patients = df_cleaned.to_dict('records')
            
            # Add status and format data for display
            logger.info("🔧 Formatting patient data...")
            for patient in patients:
                # Add status based on prior authorization
                if patient.get('prior_authorization', False):
                    patient['status'] = 'Approved'
                else:
                    patient['status'] = 'Pending'
                
                # Format claim amount
                if 'claim_amount' in patient and patient['claim_amount'] is not None:
                    patient['claim_amount'] = float(patient['claim_amount'])
                
                # Format service date
                if 'service_date' in patient and patient['service_date'] is not None:
                    try:
                        date_obj = pd.to_datetime(patient['service_date'])
                        patient['service_date'] = date_obj.strftime('%Y-%m-%d')
                    except:
                        pass
            
            logger.info(f"✅ Successfully processed {len(patients)} patients")
            return patients
            
        except Exception as e:
            logger.error(f"❌ Error in get_patients_data: {str(e)}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            raise
    
    def submit_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a new claim using the agentic system"""
        try:
            if self.agentic_available and hasattr(self, 'claim_flow') and self.claim_flow is not None:
                # Use agentic system for claim processing
                logger.info("🤖 Processing claim with agentic system...")
                
                # Format claim data for agentic processing
                formatted_claim = {
                    'patient_id': claim_data.get('patient_id'),
                    'patient_name': claim_data.get('patient_name'),
                    'procedure_code': claim_data.get('procedure_code'),
                    'diagnosis_code': claim_data.get('diagnosis_code'),
                    'claim_amount': float(claim_data.get('claim_amount', 0)),
                    'service_date': claim_data.get('service_date', datetime.now().strftime('%Y-%m-%d')),
                    'provider': claim_data.get('provider'),
                    'insurer': claim_data.get('insurer'),
                    'priority': claim_data.get('priority', 'normal'),
                    'notes': claim_data.get('notes', '')
                }
                
                # Process through agentic claim flow
                # This will use the full multi-agent system
                result = asyncio.run(self.claim_flow.process_claim(formatted_claim))
                
                return {
                    'success': True,
                    'claim_id': result.get('claim_id'),
                    'status': result.get('status', 'submitted'),
                    'message': f'Claim submitted successfully with agentic processing',
                    'processing_details': result,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Fallback submission without agentic system
                return self.submit_claim_fallback(claim_data)
                
        except Exception as e:
            logger.error(f"[ERROR] Error in agentic claim submission: {str(e)}")
            return self.submit_claim_fallback(claim_data)
    
    def submit_claim_fallback(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback claim submission without agentic system"""
        try:
            # Generate a simple claim ID
            claim_id = f"CLM-{datetime.now().strftime('%Y%m%d')}-{len(self.patients_df) + 1:04d}"
            
            # Basic validation
            required_fields = ['patient_name', 'procedure_code', 'claim_amount']
            missing_fields = [field for field in required_fields if not claim_data.get(field)]
            
            if missing_fields:
                return {
                    'success': False,
                    'error': f'Missing required fields: {", ".join(missing_fields)}'
                }
            
            # Simulate claim processing
            status = 'submitted'
            if float(claim_data.get('claim_amount', 0)) > 10000:
                status = 'requires_review'
            
            return {
                'success': True,
                'claim_id': claim_id,
                'status': status,
                'message': 'Claim submitted successfully (fallback mode)',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Claim submission failed: {str(e)}'
            }


# Initialize the API
api = HealthcareDashboardAPI()

# Routes
@app.route('/')
def dashboard():
    """Serve the main dashboard"""
    return send_from_directory(WEB_DIR, 'index.html')

@app.route('/api/metrics')
def get_metrics():
    """Get dashboard metrics"""
    metrics = api.get_dashboard_metrics()
    return jsonify(metrics)

@app.route('/api/patients')
def get_patients():
    """Get patient data"""
    try:
        logger.info("📋 API request received for patients data")
        patients = api.get_patients_data()
        logger.info(f"📋 Returning {len(patients)} patients")
        return jsonify(patients)
    except Exception as e:
        logger.error(f"❌ Error in get_patients: {str(e)}")
        logger.error(f"❌ Exception type: {type(e)}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reload-data', methods=['POST'])
def reload_data():
    """Reload data from CSV files for real-time updates"""
    try:
        api.reload_data()
        patient_count = len(api.patients_df) if api.patients_df is not None else 0
        return jsonify({
            'success': True,
            'message': f'Data reloaded successfully - {patient_count} patients',
            'patient_count': patient_count,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"[ERROR] Error reloading data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/denials')
def get_denials():
    """Get denial data (for internal use only)"""
    denials = api.denials_df.to_dict('records') if api.denials_df is not None and not api.denials_df.empty else []
    return jsonify(denials)

@app.route('/api/submit-claim', methods=['POST'])
def submit_claim():
    """Submit a new claim for processing"""
    try:
        data = request.get_json()
        logger.info(f"📋 New claim submitted: {data}")
        result = api.submit_claim(data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"[ERROR] Error submitting claim: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/patient/<patient_id>')
def get_patient_details(patient_id):
    """Get detailed patient information"""
    if api.patients_df is None or api.patients_df.empty:
        return jsonify({'error': 'No patient data available'}), 404
    
    patient = api.patients_df[api.patients_df['patient_id'] == patient_id]
    if patient.empty:
        return jsonify({'error': 'Patient not found'}), 404
    
    return jsonify(patient.to_dict('records')[0])

# Static files
@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory(WEB_DIR, filename)

@app.route('/dashboard.js')
def dashboard_js():
    """Serve dashboard JavaScript"""
    return send_from_directory(WEB_DIR, 'dashboard.js')

@app.route('/styles.css')
def styles_css():
    """Serve dashboard CSS"""
    return send_from_directory(WEB_DIR, 'styles.css')

if __name__ == '__main__':
    print("🏥 Starting Healthcare Claims Processing System...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("🔗 API endpoints:")
    print("   • /api/metrics - Dashboard metrics")
    print("   • /api/patients - Patient data")
    print("   • /api/submit-claim - Submit new claim")
    print("   • /api/patient/<id> - Get patient details")
    
    if api.agentic_available:
        print("[SUCCESS] Agentic system integration: ACTIVE")
    else:
        print("[WARNING] Agentic system integration: FALLBACK MODE")
    
    # Enable debug mode temporarily to see request logs
    app.run(debug=True, host='127.0.0.1', port=5000)
