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
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Any

# Configure logging: console + rotating file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard")
logger.setLevel(logging.INFO)

logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'logs'))
os.makedirs(logs_dir, exist_ok=True)

# Rotating app log
app_log_path = os.path.join(logs_dir, 'dashboard_app.log')
if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    file_handler = RotatingFileHandler(app_log_path, maxBytes=2_000_000, backupCount=3, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s'))
    logger.addHandler(file_handler)

# Unified JSONL execution log writer
def append_execution_log(entry: dict):
    try:
        path = os.path.join(logs_dir, 'execution_log.jsonl')
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=True) + '\n')
    except Exception:
        logger.exception("Failed writing execution log")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import centralized logger
try:
    from tools.execution_logger import execution_logger, log_execution, log_error
    HAS_EXECUTION_LOGGER = True
    logger.info("[SUCCESS] Centralized execution logger imported")
except ImportError as e:
    HAS_EXECUTION_LOGGER = False
    logger.warning(f"[WARNING] Centralized execution logger not available: {e}")

# Import user-friendly translator
try:
    from tools.user_friendly_translator import translate_to_user_friendly
    HAS_TRANSLATOR = True
    logger.info("[SUCCESS] User-friendly translator imported")
except ImportError as e:
    HAS_TRANSLATOR = False
    logger.warning(f"[WARNING] User-friendly translator not available: {e}")

# Import agentic system components
try:
    from tools.openemr_data_loader import OpenEMRPatientLoader, OpenEMRDenialLearningLoader
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
        from tools.openemr_data_loader import OpenEMRPatientLoader, OpenEMRDenialLearningLoader
        AGENTIC_AVAILABLE = True
        logger.info("[SUCCESS] OpenEMR data loaders imported successfully")
    except ImportError as e2:
        logger.error(f"[ERROR] OpenEMR data loaders not available: {e2}")
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
    append_execution_log({
        'type': 'http_request',
        'method': request.method,
        'url': request.url,
        'headers': {k: v for k, v in request.headers.items() if k.lower() in ('content-type','accept')},
        'timestamp': datetime.now().isoformat()
    })

@app.after_request
def log_response_info(response):
    logger.info(f"🌐 Response: {response.status_code}")
    append_execution_log({
        'type': 'http_response',
        'status': response.status_code,
        'url': request.url,
        'timestamp': datetime.now().isoformat()
    })
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

def sync_enhance_activity(activity_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous wrapper for creating enhanced activities with user-friendly translations
    """
    if HAS_TRANSLATOR:
        try:
            # Create event loop if none exists
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is already running, we can't use run_until_complete
                    # Fall back to synchronous translation
                    return sync_fallback_translation(activity_data)
            except RuntimeError:
                # No event loop exists, create one
                loop = None
            
            if loop and not loop.is_running():
                # Create an async function to run the translation
                async def run_translation():
                    return await translate_to_user_friendly(activity_data)
                
                # Run the async translation
                translation_result = loop.run_until_complete(run_translation())
                activity_data.update(translation_result)
                activity_data['has_translation'] = True
                return activity_data
            else:
                # Use fallback translation
                return sync_fallback_translation(activity_data)
                
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return sync_fallback_translation(activity_data)
    else:
        return sync_fallback_translation(activity_data)

def sync_fallback_translation(activity_data: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous fallback for user-friendly translations"""
    agent = activity_data.get('agent', 'System')
    activity = activity_data.get('activity', 'Processing')
    details = activity_data.get('details', '')
    
    # Agent-specific user-friendly translations
    agent_translations = {
        'Risk Predictor': {
            'activity': 'Reviewing your claim for approval likelihood',
            'details': 'Our AI system is analyzing your claim details to predict the best outcome.'
        },
        'Auto Corrector': {
            'activity': 'Checking and improving your claim details',
            'details': 'We\'re reviewing your claim information and making sure everything is complete.'
        },
        'Claim Submitter': {
            'activity': 'Submitting your claim to insurance',
            'details': 'Your claim is being sent to your insurance company for processing.'
        },
        'Appeal Generator': {
            'activity': 'Preparing an appeal for your claim',
            'details': 'We\'re creating a formal appeal letter to challenge the initial decision.'
        },
        'Resubmitter': {
            'activity': 'Resubmitting your improved claim',
            'details': 'After making corrections, we\'re sending your updated claim back to insurance.'
        },
        'Feedback Learner': {
            'activity': 'Learning from your claim outcome',
            'details': 'We\'re analyzing the results to improve our process for future claims.'
        }
    }
    
    # Get user-friendly translation or use generic
    translation = agent_translations.get(agent, {
        'activity': 'Processing your healthcare claim',
        'details': 'We\'re working on your claim to ensure the best possible outcome.'
    })
    
    # Convert technical terms in details
    user_friendly_details = details
    technical_translations = {
        'Risk Score': 'Approval likelihood',
        'Azure OpenAI Analysis': 'AI-powered review',
        'Policy Coverage': 'Insurance coverage',
        'MCP client': 'External verification',
        'Eligibility check': 'Insurance verification',
        'Prior authorization': 'Insurance pre-approval'
    }
    
    for technical, friendly in technical_translations.items():
        user_friendly_details = user_friendly_details.replace(technical, friendly)
    
    # Add user-friendly fields
    activity_data['user_friendly_activity'] = translation['activity']
    activity_data['user_friendly_details'] = user_friendly_details if user_friendly_details else translation['details']
    activity_data['next_steps'] = 'We\'ll update you as soon as this step is complete.'
    activity_data['has_translation'] = False  # Mark as fallback translation
    
    return activity_data

def parse_log_entry(log_entry, patient_id, tracking_id, activity_counter=None):
    """Parse workflow log entries into user-friendly activity descriptions"""
    try:
        if not isinstance(log_entry, str):
            return None

        activity_id = activity_counter if activity_counter else len(str(log_entry))

        # 🧠 RISK PREDICTOR AGENT ACTIVITIES
        if "[RiskPredictor" in log_entry:
            if "Risk:" in log_entry and "Confidence:" in log_entry:
                import re
                risk_match = re.search(r"Risk: ([\d.]+)", log_entry)
                confidence_match = re.search(r"Confidence: ([\d.]+)", log_entry)
                issues_match = re.search(r"Issues: (\d+)", log_entry)
                coverage_match = re.search(r"Policy Coverage: (\w+)", log_entry)
                denial_rate_match = re.search(r"Historical Denial Rate: ([\d.]+)", log_entry)

                return {
                    "id": activity_id,
                    "patient_id": patient_id,
                    "tracking_id": tracking_id,
                    "agent": "Risk Predictor",
                    "activity": "AI Risk Analysis Complete",
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                    "details": (
                        f"🧠 Azure OpenAI Analysis: Risk Score {risk_match.group(1) if risk_match else 'N/A'} "
                        f"(Confidence: {confidence_match.group(1) if confidence_match else 'N/A'}). "
                        f"Found {issues_match.group(1) if issues_match else '0'} issues. "
                        f"Policy Coverage: {coverage_match.group(1) if coverage_match else 'Unknown'}. "
                        f"Historical Denial Rate: {denial_rate_match.group(1) if denial_rate_match else '0'}. "
                        "Enhanced patient data via MCP, validated ICD/CPT codes, analyzed denial patterns."
                    ),
                    "category": "ai_risk_analysis",
                }
            elif "Gets enhanced patient data via MCP" in log_entry or "policy check" in log_entry.lower():
                return {
                    "id": activity_id,
                    "patient_id": patient_id,
                    "tracking_id": tracking_id,
                    "agent": "Risk Predictor",
                    "activity": "MCP Data Enhancement",
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                    "details": "🔍 Getting enhanced patient data via MCP, checking policy coverage, analyzing denial patterns, validating ICD/CPT.",
                    "category": "data_enhancement",
                }
            elif "Error:" in log_entry:
                return {
                    "id": activity_id,
                    "patient_id": patient_id,
                    "tracking_id": tracking_id,
                    "agent": "Risk Predictor",
                    "activity": "Risk Analysis Error",
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "details": f"❌ {log_entry.split('Error:')[-1].strip()}",
                    "category": "error",
                }

        # 🔧 AUTO CORRECTOR AGENT ACTIVITIES
        elif "[AutoCorrector" in log_entry:
            if "Applied" in log_entry and "corrections" in log_entry:
                import re
                corr_match = re.search(r"Applied (\d+) corrections", log_entry)
                count = corr_match.group(1) if corr_match else "0"
                return {
                    "id": activity_id,
                    "patient_id": patient_id,
                    "tracking_id": tracking_id,
                    "agent": "Auto Corrector",
                    "activity": "AI Data Corrections Applied",
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                    "details": f"🔧 Azure OpenAI fixed {count} data problems. Corrections for demographics, prior auth, medical history.",
                    "category": "ai_correction",
                }
            elif "[RESOLVED]" in log_entry:
                issue = log_entry.split("[RESOLVED]")[-1].strip()
                return {
                    "id": activity_id,
                    "patient_id": patient_id,
                    "tracking_id": tracking_id,
                    "agent": "Auto Corrector",
                    "activity": "Data Issue Resolved",
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                    "details": ("✅ Resolved: " + (issue[:100] + ("..." if len(issue) > 100 else ""))),
                    "category": "issue_resolution",
                }
            elif "Generated missing prior authorization" in log_entry or "prior auth" in log_entry.lower():
                return {
                    "id": activity_id,
                    "patient_id": patient_id,
                    "tracking_id": tracking_id,
                    "agent": "Auto Corrector",
                    "activity": "Prior Authorization Generated",
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                    "details": "📋 Generated missing prior authorization numbers and added medical documentation.",
                    "category": "prior_auth",
                }

        # 📤 CLAIM SUBMITTER AGENT ACTIVITIES
        elif "[ClaimSubmitter" in log_entry:
            if "Eligibility verified" in log_entry:
                return {
                    "id": activity_id,
                    "patient_id": patient_id,
                    "tracking_id": tracking_id,
                    "agent": "Claim Submitter",
                    "activity": "Real-time Eligibility Verified",
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                    "details": "✅ Eligibility confirmed via real-time MCP check.",
                    "category": "eligibility_check",
                }
            elif "Claim submitted, waiting for processing" in log_entry:
                return {
                    "id": activity_id,
                    "patient_id": patient_id,
                    "tracking_id": tracking_id,
                    "agent": "Claim Submitter",
                    "activity": "Claim Submitted to Insurance API",
                    "status": "processing",
                    "timestamp": datetime.now().isoformat(),
                    "details": "📤 Submitted to insurer API. Waiting ~60s for processing.",
                    "category": "api_submission",
                }
            elif "Eligibility check failed" in log_entry:
                reason = log_entry.split("failed:")[-1].strip() if "failed:" in log_entry else "Unknown reason"
                return {
                    "id": activity_id,
                    "patient_id": patient_id,
                    "tracking_id": tracking_id,
                    "agent": "Claim Submitter",
                    "activity": "Eligibility Check Failed",
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "details": f"❌ Eligibility verification failed: {reason}",
                    "category": "eligibility_error",
                }
            elif "Routes based on insurer" in log_entry or "routing to" in log_entry.lower():
                return {
                    "id": activity_id,
                    "patient_id": patient_id,
                    "tracking_id": tracking_id,
                    "agent": "Claim Submitter",
                    "activity": "Insurance API Routing",
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                    "details": "🔀 Routed by insurer: Primary(8081)/Secondary(8082)",
                    "category": "routing",
                }

        # 📝 APPEAL GENERATOR AGENT ACTIVITIES
        elif "[AppealGenerator" in log_entry:
            if "Appeal created" in log_entry:
                return {
                    "id": activity_id,
                    "patient_id": patient_id,
                    "tracking_id": tracking_id,
                    "agent": "Appeal Generator",
                    "activity": "AI Appeal Letter Generated",
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                    "details": "📝 Azure OpenAI created formal appeal letter and PDF.",
                    "category": "ai_appeal_generation",
                }
            elif "formal appeal letter" in log_entry.lower():
                return {
                    "id": activity_id,
                    "patient_id": patient_id,
                    "tracking_id": tracking_id,
                    "agent": "Appeal Generator",
                    "activity": "Appeal Analysis in Progress",
                    "status": "processing",
                    "timestamp": datetime.now().isoformat(),
                    "details": "🤖 Generating formal appeal with supporting docs.",
                    "category": "appeal_analysis",
                }

        # 🔄 RESUBMITTER AGENT ACTIVITIES
        elif "[Resubmitter" in log_entry:
            if "resubmitted" in log_entry.lower() or "Claim resubmitted" in log_entry:
                return {
                    "id": activity_id,
                    "patient_id": patient_id,
                    "tracking_id": tracking_id,
                    "agent": "Resubmitter",
                    "activity": "Claim Resubmitted with Appeal",
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                    "details": "🔄 Resubmitted claim with appeal packet and corrections.",
                    "category": "resubmission",
                }

        # 📈 FEEDBACK LEARNER AGENT ACTIVITIES
        elif "[FeedbackLearner" in log_entry:
            if "Learning pattern updated" in log_entry or "learned" in log_entry.lower():
                return {
                    "id": activity_id,
                    "patient_id": patient_id,
                    "tracking_id": tracking_id,
                    "agent": "Feedback Learner",
                    "activity": "AI Learning Pattern Updated",
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                    "details": "📈 Updated denial patterns and feedback logs.",
                    "category": "ai_learning",
                }
            elif "Analyzes final claim outcome" in log_entry:
                return {
                    "id": activity_id,
                    "patient_id": patient_id,
                    "tracking_id": tracking_id,
                    "agent": "Feedback Learner",
                    "activity": "Outcome Analysis in Progress",
                    "status": "processing",
                    "timestamp": datetime.now().isoformat(),
                    "details": "🔍 Analyzing outcome for future improvements.",
                    "category": "outcome_analysis",
                }

        # 🔗 MCP SERVER ACTIVITIES
        elif "MCP" in log_entry and ("get_patient_data" in log_entry or "validate_claim" in log_entry or "submit_to_insurer" in log_entry):
            tool_name = "Patient Data Retrieval" if "get_patient_data" in log_entry else (
                "Claim Validation" if "validate_claim" in log_entry else "Insurance Submission"
            )
            return {
                "id": activity_id,
                "patient_id": patient_id,
                "tracking_id": tracking_id,
                "agent": "MCP Server",
                "activity": f"{tool_name} via MCP",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "details": "🔌 MCP executed tool against insurer APIs.",
                "category": "mcp_tools",
            }

        # 🎯 ROUTING DECISIONS
        elif "routing to" in log_entry.lower():
            route_target = log_entry.split("routing to")[-1].strip()
            return {
                "id": activity_id,
                "patient_id": patient_id,
                "tracking_id": tracking_id,
                "agent": "Workflow Router",
                "activity": "Routing Decision Made",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "details": f"🎯 Intelligent routing decision: {route_target}",
                "category": "routing_decision",
            }

        # 💰 CLAIM STATUS UPDATES
        elif any(s in log_entry.lower() for s in ["approved", "denied", "pending"]):
            status_word = "approved" if "approved" in log_entry.lower() else (
                "denied" if "denied" in log_entry.lower() else "pending"
            )
            icon = "✅" if status_word == "approved" else ("❌" if status_word == "denied" else "⏳")
            return {
                "id": activity_id,
                "patient_id": patient_id,
                "tracking_id": tracking_id,
                "agent": "Insurance API",
                "activity": f"Claim Status: {status_word.title()}",
                "status": status_word,
                "timestamp": datetime.now().isoformat(),
                "details": f"{icon} Insurance API response: {log_entry[:100]}{'...' if len(log_entry) > 100 else ''}",
                "category": "claim_status",
            }

        # 📊 GENERAL WORKFLOW STEPS
        elif "[" in log_entry and "]" in log_entry:
            agent_match = log_entry.split("]")[0].replace("[", "")
            return {
                "id": activity_id,
                "patient_id": patient_id,
                "tracking_id": tracking_id,
                "agent": agent_match,
                "activity": "Workflow Step",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "details": log_entry[:150] + "..." if len(log_entry) > 150 else log_entry,
                "category": "workflow_step",
            }

        return None

    except Exception:
        logger.exception("Error parsing log entry")
        return None

class HealthcareDashboardAPI:
    """Professional Healthcare Dashboard API integrated with Agentic System"""
    
    def __init__(self):
        self.patients_df = None
        self.denials_df = None
        # Initialize active processing tracker for real-time activity (session_id -> info)
        self.active_processing = {}
        # Initialize completed activities history (real activities only - start empty)
        self.completed_activities = []
        
        # Clear any old execution logs to start fresh
        self._clear_old_logs()

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

                # Initialize data loaders - Try OpenEMR first, fallback to CSV
                data_source = getattr(Settings, 'DATA_SOURCE', 'openemr')  # Prefer OpenEMR
                self.data_source = data_source

                # Try to initialize OpenEMR database loaders
                try:
                    from tools.openemr_data_loader import OpenEMRPatientLoader, OpenEMRDenialLearningLoader
                    self.patient_loader = OpenEMRPatientLoader()
                    self.denial_loader = OpenEMRDenialLearningLoader()
                    logger.info("[SUCCESS] Initialized OpenEMR database data loaders")
                except Exception as openemr_e:
                    logger.warning(f"[WARNING] ⚠️ OpenEMR loaders failed, will use CSV fallback: {openemr_e}")
                    self.patient_loader = None
                    self.denial_loader = None

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
                # Set loaders to None so load_data can handle fallback
                self.patient_loader = None
                self.denial_loader = None
        else:
            self.agentic_available = False
            # Set loaders to None so load_data can handle fallback
            self.patient_loader = None
            self.denial_loader = None

        self.load_data()
    
    def _clear_old_logs(self):
        """Clear ALL old log files on startup for completely clean state"""
        try:
            # List of all log files to clear for fresh start
            log_files_to_clear = [
                'execution_log.jsonl',
                'execution_trace.jsonl',
                'claim_flow_log.jsonl',
                'risk_predictor_log.jsonl',
                'dashboard_app.log',
                'api_server.log',
                'claim_flow.log',
                'risk_predictor.log',
                'appeal_generator.log',
                'auto_corrector.log',
                'claim_submitter.log',
                'feedback_learner.log',
                'resubmitter.log',
                'insurer_api.log',
                'mcp_client.log',
                'execution_trace.log'
            ]
            
            cleared_count = 0
            for log_file in log_files_to_clear:
                log_path = os.path.join(logs_dir, log_file)
                if os.path.exists(log_path):
                    with open(log_path, 'w', encoding='utf-8') as f:
                        f.write('')  # Clear the file completely
                    cleared_count += 1
            
            logger.info(f"🧹 Cleared {cleared_count} old log files for completely fresh start")
            logger.info("✨ System starting with clean state - NO fake activities")
        except Exception as e:
            logger.warning(f"Could not clear old logs: {e}")
    
    def load_data(self):
        """Load patient and denial data - Priority: OpenEMR database, Fallback: CSV files"""
        openemr_success = False
        
        # FIRST PRIORITY: Try OpenEMR database
        try:
            if self.agentic_available and hasattr(self, 'patient_loader'):
                # Check if OpenEMR data loader has actual data
                if self.patient_loader.patients_df is not None and not self.patient_loader.patients_df.empty:
                    self.patients_df = self.patient_loader.patients_df
                    logger.info(f"[SUCCESS] ✅ Loaded {len(self.patients_df)} patients from OpenEMR database")
                    openemr_success = True
                    
                    # Handle denial loader carefully
                    try:
                        self.denials_df = self.denial_loader.denial_df
                    except AttributeError:
                        logger.warning("[WARNING] denial_df not found, using alternative approach")
                        self.denials_df = getattr(self.denial_loader, 'denials_df', pd.DataFrame())
                else:
                    logger.warning("[WARNING] ⚠️ OpenEMR database returned empty data - will try CSV fallback")
        except Exception as e:
            logger.warning(f"[WARNING] ⚠️ OpenEMR database failed: {e}")
        
        # FALLBACK: Try CSV files if OpenEMR failed or returned empty data
        if not openemr_success:
            logger.info("[INFO] 🔄 Attempting CSV fallback...")
            try:
                # Try to import CSV data loader
                from tools.csv_data_loader import PatientLoader, DenialLearningLoader
                
                # Try different CSV file paths
                csv_paths_to_try = [
                    "data/patients1.csv",  # Found in attachments
                    "data/patients.csv",   # Standard path
                    os.path.join(DATA_DIR, "patients1.csv"),  # Absolute path
                    os.path.join(DATA_DIR, "patients.csv")    # Absolute path
                ]
                
                csv_success = False
                for csv_path in csv_paths_to_try:
                    try:
                        if os.path.exists(csv_path):
                            logger.info(f"[INFO] 📄 Trying CSV file: {csv_path}")
                            csv_patient_loader = PatientLoader(csv_path)
                            
                            if csv_patient_loader.patients_df is not None and not csv_patient_loader.patients_df.empty:
                                self.patients_df = csv_patient_loader.patients_df
                                logger.info(f"[SUCCESS] ✅ Loaded {len(self.patients_df)} patients from CSV: {csv_path}")
                                csv_success = True
                                
                                # Load denial data from CSV
                                try:
                                    csv_denial_loader = DenialLearningLoader()
                                    self.denials_df = csv_denial_loader.denial_df
                                    logger.info(f"[SUCCESS] ✅ Loaded denial data from CSV")
                                except Exception as denial_e:
                                    logger.warning(f"[WARNING] Could not load denial CSV: {denial_e}")
                                    self.denials_df = pd.DataFrame()
                                
                                break
                        else:
                            logger.debug(f"[DEBUG] CSV file not found: {csv_path}")
                    except Exception as csv_e:
                        logger.debug(f"[DEBUG] Failed to load CSV {csv_path}: {csv_e}")
                        continue
                
                if not csv_success:
                    logger.error("[ERROR] ❌ All CSV fallback attempts failed")
                    self.patients_df = pd.DataFrame()
                    self.denials_df = pd.DataFrame()
                    
            except ImportError as import_e:
                logger.error(f"[ERROR] ❌ Could not import CSV loader: {import_e}")
                self.patients_df = pd.DataFrame()
                self.denials_df = pd.DataFrame()
            except Exception as e:
                logger.error(f"[ERROR] ❌ CSV fallback failed: {e}")
                self.patients_df = pd.DataFrame()
                self.denials_df = pd.DataFrame()
        
        # Final status report
        patient_count = len(self.patients_df) if self.patients_df is not None else 0
        denial_count = len(self.denials_df) if self.denials_df is not None else 0
        
        if patient_count > 0:
            source = "OpenEMR database" if openemr_success else "CSV files"
            logger.info(f"[FINAL] 🎯 Data loading complete: {patient_count} patients, {denial_count} denials from {source}")
        else:
            logger.error("[FINAL] ❌ No patient data loaded from any source")
    
    def reload_data(self):
        """Reload data with fallback mechanism - Priority: OpenEMR database, Fallback: CSV files"""
        logger.debug(f"🔄 Checking for database updates...")
        
        # Store previous count for comparison
        previous_count = len(self.patients_df) if self.patients_df is not None else 0
        openemr_success = False
        
        # FIRST PRIORITY: Try OpenEMR database
        try:
            # Always try OpenEMR database first for real-time updates
            from tools.openemr_data_loader import OpenEMRPatientLoader, OpenEMRDenialLearningLoader
            
            self.patient_loader = OpenEMRPatientLoader()
            self.denial_loader = OpenEMRDenialLearningLoader()
            
            # Force reload from database
            self.patient_loader.reload_data()
            
            # Check if we got valid data
            if self.patient_loader.patients_df is not None and not self.patient_loader.patients_df.empty:
                self.patients_df = self.patient_loader.patients_df
                current_count = len(self.patients_df)
                
                # Only log if count changed to reduce spam
                if current_count != previous_count:
                    logger.info(f"[SUCCESS] ✅ Data updated: {current_count} patients from OpenEMR database")
                else:
                    logger.debug(f"[DEBUG] No data changes - {current_count} patients from OpenEMR")
                    
                openemr_success = True
            else:
                logger.warning("[WARNING] ⚠️ OpenEMR database reload returned empty data - trying CSV fallback")
                    
        except Exception as e:
            logger.warning(f"[WARNING] ⚠️ Error reloading from OpenEMR: {e}")
        
        # FALLBACK: Try CSV files if OpenEMR failed
        if not openemr_success:
            logger.info("[INFO] 🔄 Attempting CSV reload fallback...")
            try:
                from tools.csv_data_loader import PatientLoader, DenialLearningLoader
                
                # Try different CSV file paths
                csv_paths_to_try = [
                    "data/patients1.csv",
                    "data/patients.csv", 
                    os.path.join(DATA_DIR, "patients1.csv"),
                    os.path.join(DATA_DIR, "patients.csv")
                ]
                
                csv_success = False
                for csv_path in csv_paths_to_try:
                    try:
                        if os.path.exists(csv_path):
                            csv_patient_loader = PatientLoader(csv_path)
                            csv_patient_loader.load_data()  # Force reload
                            
                            if csv_patient_loader.patients_df is not None and not csv_patient_loader.patients_df.empty:
                                self.patients_df = csv_patient_loader.patients_df
                                current_count = len(self.patients_df)
                                
                                if current_count != previous_count:
                                    logger.info(f"[SUCCESS] ✅ Data updated: {current_count} patients from CSV: {csv_path}")
                                else:
                                    logger.debug(f"[DEBUG] No data changes - {current_count} patients from CSV")
                                
                                csv_success = True
                                break
                    except Exception as csv_e:
                        logger.debug(f"[DEBUG] Failed to reload CSV {csv_path}: {csv_e}")
                        continue
                
                if not csv_success:
                    logger.error("[ERROR] ❌ All CSV reload attempts failed")
                    
            except Exception as e:
                logger.error(f"[ERROR] ❌ CSV reload fallback failed: {e}")
        
        # Final status
        final_count = len(self.patients_df) if self.patients_df is not None else 0
        source = "OpenEMR database" if openemr_success else "CSV files"
        logger.debug(f"[FINAL] 🎯 Reload complete: {final_count} patients from {source}")

    def clear_activity_and_logs(self) -> Dict[str, Any]:
        """Admin helper: wipe in-memory activities and truncate logs."""
        try:
            cleared = { 'activities': 0, 'files': 0 }
            cleared['activities'] = len(self.completed_activities)
            self.completed_activities = []
            self.active_processing = {}
            # Truncate log files
            log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log') or f.endswith('.jsonl')]
            for name in log_files:
                path = os.path.join(logs_dir, name)
                try:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write('')
                    cleared['files'] += 1
                except Exception:
                    pass
            return { 'success': True, 'cleared': cleared, 'timestamp': datetime.now().isoformat() }
        except Exception as e:
            return { 'success': False, 'error': str(e) }
    
    def generate_sample_data(self):
        """Generate sample patient data for demonstration when OpenEMR is unavailable"""
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
    
    def get_real_time_agent_activity(self) -> List[Dict[str, Any]]:
        """Get real-time agent activity — reads from MCP agent log files + in-memory sessions."""
        activities = []
        current_time = datetime.now()

        # ── 1. In-memory active processing sessions (live, highest priority) ──
        if hasattr(self, 'active_processing') and self.active_processing:
            agent_labels = {
                'risk_predictor': ('Risk Predictor', '🧠 Analyzing medical risk'),
                'auto_corrector': ('Auto Corrector', '🔧 Fixing claim data issues'),
                'claim_submitter': ('Claim Submitter', '📤 Submitting to insurance'),
                'appeal_generator': ('Appeal Generator', '📝 Generating appeal letter'),
                'resubmitter': ('Resubmitter', '🔄 Resubmitting corrected claim'),
                'feedback_learner': ('Feedback Learner', '📈 Learning from outcome'),
            }
            for session_id, session_data in self.active_processing.items():
                patient_name = session_data.get('patient_name', 'Unknown Patient')
                agent_key = session_data.get('current_agent', 'unknown')
                label, desc = agent_labels.get(agent_key, (agent_key, f'Processing {patient_name}'))
                start_time = session_data.get('start_time', current_time)
                activities.append({
                    'id': session_id,
                    'agent': label,
                    'activity': f'{desc} for {patient_name}',
                    'user_friendly_activity': f'{desc} for {patient_name}',
                    'patient_id': session_data.get('patient_id', ''),
                    'patient_name': patient_name,
                    'duration': int((current_time - start_time).total_seconds()),
                    'status': 'processing',
                    'timestamp': start_time.isoformat(),
                    'details': f'Active session started {int((current_time - start_time).total_seconds())}s ago',
                    'category': 'live',
                })

        # ── 2. Read from MCP agent JSONL log files (persistent across restarts) ──
        mcp_log_files = [
            ('RiskPredictor-MCP_log.jsonl', 'Risk Predictor'),
            ('AutoCorrector-MCP_log.jsonl', 'Auto Corrector'),
            ('ClaimSubmitter-MCP_log.jsonl', 'Claim Submitter'),
            ('AppealGenerator_log.jsonl', 'Appeal Generator'),
            ('resubmitter_log.jsonl', 'Resubmitter'),
            ('feedback_learner_log.jsonl', 'Feedback Learner'),
        ]
        agent_icons = {
            'Risk Predictor': '🧠', 'Auto Corrector': '🔧',
            'Claim Submitter': '📤', 'Appeal Generator': '📝',
            'Resubmitter': '🔄', 'Feedback Learner': '📈',
        }
        for log_file, agent_label in mcp_log_files:
            log_path = os.path.join(logs_dir, log_file)
            if not os.path.exists(log_path):
                continue
            try:
                with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
                # Read last 20 lines per file
                for line in lines[-20:]:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    ts = entry.get('timestamp', current_time.isoformat())
                    state = entry.get('state_snapshot', {})
                    claim_id = entry.get('claim_id') or state.get('claim_id', '')
                    patient_id = state.get('patient_id', '')
                    risk_score = state.get('risk_score')
                    issues = state.get('issues', [])
                    dq = state.get('data_quality_score', 0)
                    final_status = state.get('final_status', '')
                    icon = agent_icons.get(agent_label, '⚙️')

                    # Build a rich detail string
                    detail_parts = []
                    if claim_id:
                        detail_parts.append(f'Claim: {claim_id}')
                    if risk_score is not None:
                        detail_parts.append(f'Risk: {int(float(risk_score)*100)}%')
                    if dq:
                        detail_parts.append(f'Data Quality: {dq}%')
                    if issues:
                        detail_parts.append(f'{len(issues)} issues found')
                    if final_status:
                        detail_parts.append(f'Status: {final_status}')
                    details = ' | '.join(detail_parts) if detail_parts else entry.get('details', '')

                    activity_text = f'{icon} {agent_label}: {entry.get("action", "Processing")}'
                    activities.append({
                        'id': f'{log_file}-{ts}',
                        'agent': agent_label,
                        'activity': activity_text,
                        'user_friendly_activity': activity_text,
                        'patient_id': patient_id or claim_id,
                        'status': 'completed' if final_status else 'completed',
                        'timestamp': ts,
                        'details': details,
                        'user_friendly_details': details,
                        'category': agent_label.lower().replace(' ', '_'),
                        'duration': 0,
                    })
            except Exception as e:
                logger.warning(f'Could not read {log_file}: {e}')

        # ── 3. Also read from claim_status.json for a summary of pipeline outcomes ──
        try:
            claim_status_path = os.path.join(DATA_DIR, 'claim_status.json')
            if os.path.exists(claim_status_path):
                with open(claim_status_path, 'r', encoding='utf-8') as f:
                    claim_statuses = json.load(f)
                status_icons = {
                    'approved': '✅', 'resubmission': '🔄',
                    'appeal_resubmitted': '📨', 'appeal_generated': '📝',
                    'appeal_resubmitted_low_confidence': '⚠️',
                    'denied': '❌', 'rejected': '❌',
                }
                for pid, entry in list(claim_statuses.items())[-8:]:
                    status = entry.get('status', 'unknown')
                    icon = status_icons.get(status, '📋')
                    ts = entry.get('updated_at') or entry.get('timestamp', current_time.isoformat())
                    claim_id = entry.get('claim_id', '')
                    risk = entry.get('risk_score', 0)
                    activities.append({
                        'id': f'status-{pid}',
                        'agent': 'Pipeline',
                        'activity': f'{icon} {pid}: {status.replace("_", " ").title()}',
                        'user_friendly_activity': f'{icon} {pid}: Claim {status.replace("_", " ").title()}',
                        'patient_id': pid,
                        'status': 'approved' if status == 'approved' else ('processing' if 'resubmit' in status else 'completed'),
                        'timestamp': ts,
                        'details': f'Claim {claim_id} | Risk: {int(float(risk)*100)}%' if claim_id else f'Risk: {int(float(risk)*100)}%',
                        'user_friendly_details': f'Claim ID: {claim_id} | AI Risk Score: {int(float(risk)*100)}%',
                        'category': 'claim_status',
                        'duration': 0,
                    })
        except Exception as e:
            logger.warning(f'Could not read claim_status for activity: {e}')

        if not activities:
            logger.info("No agent activities found in logs or memory")
            return []

        # Sort newest first, deduplicate by id, limit to 25
        seen = set()
        unique = []
        for a in sorted(activities, key=lambda x: x.get('timestamp', ''), reverse=True):
            if a['id'] not in seen:
                seen.add(a['id'])
                unique.append(a)
        result = unique[:25]
        logger.info(f"Returning {len(result)} agent activities")
        return result
    
    def clear_old_activities(self):
        """Clear activities older than 10 minutes to prevent memory buildup"""
        if hasattr(self, 'completed_activities') and self.completed_activities:
            cutoff_time = datetime.now() - timedelta(minutes=10)
            original_count = len(self.completed_activities)
            self.completed_activities = [
                activity for activity in self.completed_activities
                if datetime.fromisoformat(activity['timestamp'].replace('Z', '+00:00')) > cutoff_time
            ]
            cleaned_count = original_count - len(self.completed_activities)
            if cleaned_count > 0:
                logger.info(f"🧹 Cleaned {cleaned_count} old activities from memory")
    
    def force_clear_all_activities(self):
        """Force clear all activities - for testing/reset purposes"""
        if hasattr(self, 'completed_activities'):
            self.completed_activities = []
        if hasattr(self, 'active_processing'):
            self.active_processing = {}
        logger.info("🧹 Force cleared all activities")
    
    def submit_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a new claim using the agentic system"""
        try:
            # Generate claim_id if not provided
            claim_id = claim_data.get('claim_id')
            if not claim_id:
                claim_id = f"CLM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{claim_data.get('patient_id', 'UNK')}"
                claim_data['claim_id'] = claim_id
            
            # Log claim submission start
            if HAS_EXECUTION_LOGGER:
                log_execution('api_server', 'CLAIM_SUBMIT_START', {
                    'claim_id': claim_id,
                    'patient_name': claim_data.get('patient_name'),
                    'procedure_code': claim_data.get('procedure_code'),
                    'claim_amount': claim_data.get('claim_amount')
                })
            
            if self.agentic_available and hasattr(self, 'claim_flow') and self.claim_flow is not None:
                # Use agentic system for claim processing
                logger.info("🤖 Processing claim with agentic system...")
                
                # Enrich claim with full patient CSV data (dob, gender, address, npi, etc.)
                from tools.csv_data_loader import patient_loader as _pl
                _pid = claim_data.get('patient_id', '')
                _csv = _pl.get_patient_by_id(_pid) or {}

                # Format claim data for agentic processing — all fields needed for X12 837P
                formatted_claim = {
                    'claim_id': claim_id,
                    'patient_id': _pid,
                    'patient_name': claim_data.get('patient_name') or _csv.get('name', ''),
                    'procedure_code': claim_data.get('procedure_code') or _csv.get('procedure_code', '99213'),
                    'diagnosis_code': claim_data.get('diagnosis_code') or _csv.get('diagnosis_code', 'Z00.00'),
                    'icd_code':       claim_data.get('diagnosis_code') or _csv.get('diagnosis_code', 'Z00.00'),
                    'cpt_code':       claim_data.get('procedure_code') or _csv.get('procedure_code', '99213'),
                    'claim_amount':   float(claim_data.get('claim_amount') or _csv.get('claim_amount', 0)),
                    'service_date':   claim_data.get('service_date') or _csv.get('service_date', datetime.now().strftime('%Y-%m-%d')),
                    'provider':       claim_data.get('provider') or _csv.get('provider', ''),
                    'insurer':        claim_data.get('insurer') or _csv.get('insurer', ''),
                    'insurance_company': claim_data.get('insurer') or _csv.get('insurer', ''),
                    # Full patient demographics for X12 837P
                    'dob':            _csv.get('dob', '1980-01-01'),
                    'gender':         _csv.get('gender', 'M'),
                    'address':        _csv.get('address', '123 MAIN ST'),
                    'phone':          _csv.get('phone', ''),
                    'email':          _csv.get('email', ''),
                    'medical_history': _csv.get('medical_history', ''),
                    'allergies':      _csv.get('allergies', ''),
                    'medications':    _csv.get('medications', ''),
                    'prior_auth':     _csv.get('prior_authorization', False),
                    # Provider fields for X12 837P
                    'provider_npi':   claim_data.get('provider_npi', '1234567890'),
                    'provider_tax_id': claim_data.get('provider_tax_id', '123456789'),
                    'insurance_id':   _pid,
                    'priority':       claim_data.get('priority', 'normal'),
                    'notes':          claim_data.get('notes', ''),
                }
                
                # Log formatted claim data
                if HAS_EXECUTION_LOGGER:
                    log_execution('api_server', 'CLAIM_FORMATTED', formatted_claim)
                
                # Set start time and create session
                start_time = datetime.now()
                # Mark processing as active (create a session entry)
                session_id = f"sess_{datetime.now().timestamp()}"
                self.active_processing[session_id] = {
                    "patient_name": formatted_claim.get("patient_name"),
                    "current_agent": "risk_predictor",
                    "start_time": start_time,
                    "status": "processing",
                }
                
                # Add initial activity entry
                patient_name = claim_data.get('patient_name', 'Unknown Patient')
                claim_amount = claim_data.get('claim_amount', 0)
                procedure_code = claim_data.get('procedure_code', 'N/A')
                
                initial_activity = {
                    'id': len(self.completed_activities) + 1,
                    'patient_name': patient_name,
                    'activity': f'Starting claim processing for {procedure_code}',
                    'status': 'processing',
                    'timestamp': start_time.isoformat(),
                    'agent': 'System',
                    'claim_amount': f'${claim_amount}',
                    'details': f'Initiating multi-agent workflow for claim processing'
                }
                self.completed_activities.append(initial_activity)
                
                # Process through agentic claim flow
                # This will use the full multi-agent system
                append_execution_log({'type': 'claim_submit_start', 'payload': formatted_claim, 'timestamp': datetime.now().isoformat()})
                result = asyncio.run(self.claim_flow.process_claim(formatted_claim))
                
                # Extract and parse workflow logs from the result
                workflow_activities = []
                if result and "workflow_log" in result:
                    logger.info(f"📝 Processing {len(result['workflow_log'])} workflow log entries")
                    for i, log_entry in enumerate(result["workflow_log"]):
                        # Parse log entries to extract agent activities
                        if isinstance(log_entry, str):
                            activity = parse_log_entry(log_entry, formatted_claim['patient_id'], 
                                                     f"track_{start_time.timestamp()}", i + 1)
                            if activity:
                                # Enhance activity with user-friendly translations
                                enhanced_activity = sync_enhance_activity(activity)
                                workflow_activities.append(enhanced_activity)
                                # Add to completed activities for real-time display
                                self.completed_activities.append(enhanced_activity)
                                logger.info(f"   📋 Added activity: {enhanced_activity['agent']} - {enhanced_activity['user_friendly_activity']}")
                                append_execution_log({'type': 'parsed_activity', 'activity': enhanced_activity, 'timestamp': datetime.now().isoformat()})
                else:
                    logger.warning("⚠️ No workflow_log found in result or result is None")
                    logger.info(f"Result keys: {list(result.keys()) if result else 'No result'}")
                    append_execution_log({'type': 'claim_submit_no_workflow_log', 'result_keys': list(result.keys()) if result else [], 'timestamp': datetime.now().isoformat()})
                    # Do NOT add any synthetic activities here. Keep feed clean until real steps exist.
                
                # Mark processing as complete and add final activity
                # Clear active session
                if session_id in self.active_processing:
                    del self.active_processing[session_id]
                end_time = datetime.now()
                processing_duration = (end_time - start_time).total_seconds()
                
                # Only append a final summary if there were real workflow steps and no error
                if len(workflow_activities) > 0 and result.get('status') not in ('error', None):
                    final_activity = {
                        'id': len(self.completed_activities) + 1,
                        'patient_name': patient_name,
                        'patient_id': formatted_claim['patient_id'],
                        'activity': 'Claim processing completed',
                        'status': result.get('status', 'completed'),
                        'timestamp': end_time.isoformat(),
                        'agent': 'System',
                        'claim_amount': f'${claim_amount}',
                        'details': f'Processing completed in {processing_duration:.1f}s. Final status: {result.get("status", "submitted")}. Total workflow steps: {len(workflow_activities)}',
                        'category': 'completion'
                    }
                    # Enhance final activity with user-friendly translations
                    enhanced_final_activity = sync_enhance_activity(final_activity)
                    self.completed_activities.append(enhanced_final_activity)
                elif result.get('status') == 'error':
                    # Add concise error activity only, no fake steps
                    error_activity = {
                        'id': len(self.completed_activities) + 1,
                        'patient_name': patient_name,
                        'patient_id': formatted_claim.get('patient_id'),
                        'agent': 'System',
                        'activity': 'Claim processing failed',
                        'status': 'error',
                        'timestamp': end_time.isoformat(),
                        'details': result.get('error', 'Unknown error'),
                        'category': 'error'
                    }
                    self.completed_activities.append(error_activity)
                append_execution_log({'type': 'claim_submit_complete', 'result': result, 'duration_sec': processing_duration, 'timestamp': datetime.now().isoformat()})
                
                logger.info(f"🎯 Claim processing completed with {len(workflow_activities)} workflow activities logged")
                
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
            # Ensure we clean up processing state on error
            # Remove all active sessions for safety
            self.active_processing.clear()
            logger.error(f"[ERROR] Error in agentic claim submission: {str(e)}")
            append_execution_log({'type': 'claim_submit_error', 'error': str(e), 'timestamp': datetime.now().isoformat()})
            
            # Add error activity if we were tracking
            if hasattr(self, 'completed_activities'):
                error_activity = {
                    'id': len(self.completed_activities) + 1,
                    'patient_name': claim_data.get('patient_name', 'Unknown Patient'),
                    'activity': 'Claim processing failed',
                    'status': 'error',
                    'timestamp': datetime.now().isoformat(),
                    'agent': 'System',
                    'claim_amount': f'${claim_data.get("claim_amount", 0)}',
                    'details': f'Error during processing: {str(e)}'
                }
                self.completed_activities.append(error_activity)
            
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
        # Store previous count
        previous_count = len(api.patients_df) if api.patients_df is not None else 0
        
        api.reload_data()
        patient_count = len(api.patients_df) if api.patients_df is not None else 0
        
        # Different message based on whether data changed
        if patient_count != previous_count:
            message = f'Data updated successfully - {patient_count} patients'
        else:
            message = f'Data checked - {patient_count} patients (no changes)'
            
        return jsonify({
            'success': True,
            'message': message,
            'patient_count': patient_count,
            'data_changed': patient_count != previous_count,
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

@app.route('/api/admin/clear-activity', methods=['POST'])
def admin_clear_activity():
    """Clear activity feed (memory) and truncate all log files."""
    result = api.clear_activity_and_logs()
    status = 200 if result.get('success') else 500
    return jsonify(result), status

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

@app.route('/api/claim-x12/<patient_id>')
def get_claim_x12(patient_id):
    """Return the raw X12 837P transaction for a patient's latest claim.
    If not stored, generate it on-the-fly from patient + claim data."""
    try:
        claim_status_path = os.path.join(DATA_DIR, 'claim_status.json')
        if not os.path.exists(claim_status_path):
            return jsonify({'error': 'No claim data found'}), 404
        with open(claim_status_path, 'r') as f:
            all_claims = json.load(f)
        patient_claims = [c for c in all_claims.values() if c.get('patient_id') == patient_id]
        if not patient_claims:
            return jsonify({'error': 'No claim found for patient'}), 404
        latest = sorted(patient_claims, key=lambda c: c.get('processing_time', 0), reverse=True)[0]

        # Return stored X12 if available
        x12 = latest.get('x12_837p')
        if x12:
            return jsonify({'claim_id': latest.get('claim_id'), 'x12_837p': x12})

        # Generate on-the-fly from patient CSV + claim data
        try:
            from tools.x12_837p_builder import build_837p
            # Look up patient from the DataFrame
            patient_data = {}
            if api.patients_df is not None and not api.patients_df.empty:
                rows = api.patients_df[api.patients_df['patient_id'] == patient_id]
                if not rows.empty:
                    patient_data = rows.iloc[0].where(rows.iloc[0].notna(), other='').to_dict()
            x12 = build_837p({
                'claim_id':       latest.get('claim_id', ''),
                'patient_id':     patient_id,
                'patient_name':   patient_data.get('name', ''),
                'dob':            patient_data.get('dob', '1980-01-01'),
                'gender':         patient_data.get('gender', 'M'),
                'insurer':        patient_data.get('insurer', ''),
                'insurance_company': patient_data.get('insurer', ''),
                'procedure_code': patient_data.get('procedure_code', '99213'),
                'diagnosis_code': patient_data.get('diagnosis_code', 'Z00.00'),
                'claim_amount':   float(patient_data.get('claim_amount', 0) or 0),
                'service_date':   patient_data.get('service_date', ''),
                'provider_npi':   patient_data.get('provider_npi', '1234567890'),
                'provider_tax_id': '123456789',
                'treatment_date': patient_data.get('service_date', ''),
            })
            # Cache it back into claim_status.json for next time
            all_claims[patient_id]['x12_837p'] = x12
            with open(claim_status_path, 'w') as f:
                json.dump(all_claims, f, indent=2)
            return jsonify({'claim_id': latest.get('claim_id'), 'x12_837p': x12})
        except Exception as gen_err:
            return jsonify({'error': f'Could not generate X12: {gen_err}'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/patient/<patient_id>')
def get_patient_details(patient_id):
    """Get detailed patient information"""
    if api.patients_df is None or api.patients_df.empty:
        return jsonify({'error': 'No patient data available'}), 404
    
    patient = api.patients_df[api.patients_df['patient_id'] == patient_id]
    if patient.empty:
        return jsonify({'error': 'Patient not found'}), 404
    
    return jsonify(patient.to_dict('records')[0])

@app.route('/api/agent-activity')
def get_agent_activity():
    """Get real-time agent activity for dashboard"""
    try:
        # Auto-clean old activities on each request
        api.clear_old_activities()
        activities = api.get_real_time_agent_activity()
        return jsonify({
            'success': True,
            'activities': activities,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"[ERROR] Error getting agent activity: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clear-activities', methods=['POST'])
def clear_activities():
    """Clear all activities - for reset/testing"""
    try:
        api.force_clear_all_activities()
        return jsonify({
            'success': True,
            'message': 'All activities cleared',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"[ERROR] Error clearing activities: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

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
    app.run(debug=False, host='0.0.0.0', port=5000)
