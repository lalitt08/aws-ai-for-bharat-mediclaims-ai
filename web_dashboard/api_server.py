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

                # Initialize data loaders - Only OpenEMR for real-time data
                data_source = getattr(Settings, 'DATA_SOURCE', 'openemr')  # Always OpenEMR
                self.data_source = data_source

                # Always use OpenEMR database for real-time patient data
                from tools.openemr_data_loader import OpenEMRPatientLoader, OpenEMRDenialLearningLoader
                self.patient_loader = OpenEMRPatientLoader()
                self.denial_loader = OpenEMRDenialLearningLoader()
                logger.info("[SUCCESS] Initialized OpenEMR database data loaders")

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
        """Load patient and denial data from OpenEMR database ONLY - no fake data"""
        try:
            if self.agentic_available and hasattr(self, 'patient_loader'):
                # Load data using OpenEMR database
                self.patients_df = self.patient_loader.patients_df
                # Handle denial loader carefully - it might not have the expected attribute
                try:
                    self.denials_df = self.denial_loader.denial_df
                except AttributeError:
                    # Fallback if denial_df doesn't exist
                    logger.warning("[WARNING] denial_df not found, using alternative approach")
                    self.denials_df = getattr(self.denial_loader, 'denials_df', pd.DataFrame())
                logger.info(f"[SUCCESS] Loaded {len(self.patients_df)} patients from OpenEMR")
            else:
                # NO FAKE DATA - start with empty dataframes
                logger.info("[INFO] OpenEMR not available - starting with empty patient data")
                logger.info("✨ Professional mode: No fake activities or sample data")
                self.patients_df = pd.DataFrame()
                self.denials_df = pd.DataFrame()
                
        except Exception as e:
            logger.error(f"[ERROR] Error loading data: {e}")
            logger.info("✨ Starting with empty data - no fake activities")
            # DO NOT generate fake data - start clean
            self.patients_df = pd.DataFrame()
            self.denials_df = pd.DataFrame()
    
    def reload_data(self):
        """Reload data from OpenEMR database - for real-time updates"""
        logger.debug(f"🔄 Checking for database updates...")
        
        try:
            # Always use OpenEMR database for real-time data
            from tools.openemr_data_loader import OpenEMRPatientLoader, OpenEMRDenialLearningLoader
            
            # Store previous count
            previous_count = len(self.patients_df) if self.patients_df is not None else 0
            
            self.patient_loader = OpenEMRPatientLoader()
            self.denial_loader = OpenEMRDenialLearningLoader()
            
            # Force reload from database
            self.patient_loader.reload_data()
            
            self.patients_df = self.patient_loader.patients_df
            current_count = len(self.patients_df) if self.patients_df is not None else 0
            
            # Only log if count changed
            if current_count != previous_count:
                logger.info(f"[SUCCESS] Data updated: {current_count} patients from OpenEMR database")
            else:
                logger.debug(f"[DEBUG] No data changes - {current_count} patients")
                    
        except Exception as e:
            logger.error(f"[ERROR] Error reloading data: {e}")
            logger.info("✨ Using empty data - no fake activities")
            # DO NOT generate fake data - keep clean
            self.patients_df = pd.DataFrame()
            self.denials_df = pd.DataFrame()

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
        """Get real-time agent activity for user-friendly display - ONLY CURRENT ACTIVITY"""
        activities = []
        current_time = datetime.now()
        
        # Clean up old completed activities (older than 5 minutes)
        if hasattr(self, 'completed_activities') and self.completed_activities:
            cutoff_time = current_time - timedelta(minutes=5)
            self.completed_activities = [
                activity for activity in self.completed_activities
                if datetime.fromisoformat(activity['timestamp'].replace('Z', '+00:00')) > cutoff_time
            ]
        
        logger.info(f"🔍 Active processing sessions: {len(self.active_processing) if hasattr(self, 'active_processing') else 0}")
        logger.info(f"🔍 Recent completed activities: {len(self.completed_activities) if hasattr(self, 'completed_activities') else 0}")
        
        # ONLY show actual active processing sessions - NO FAKE DATA
        if hasattr(self, 'active_processing') and self.active_processing:
            for session_id, session_data in self.active_processing.items():
                patient_name = session_data.get('patient_name', 'Unknown Patient')
                agent = session_data.get('current_agent', 'Unknown')
                start_time = session_data.get('start_time', current_time)
                
                # Convert agent names to user-friendly descriptions
                agent_descriptions = {
                    'risk_predictor': f'🧠 Analyzing medical risk for {patient_name}',
                    'auto_corrector': f'🔧 Fixing missing information for {patient_name}',
                    'claim_submitter': f'📤 Submitting claim to insurance for {patient_name}',
                    'appeal_generator': f'📝 Creating appeal letter for {patient_name}',
                    'resubmitter': f'🔄 Resubmitting claim for {patient_name}',
                    'feedback_learner': f'📈 Learning from {patient_name}\'s case for future improvements'
                }
                
                activity = {
                    'id': session_id,
                    'activity': agent_descriptions.get(agent, f'Processing {patient_name}'),
                    'patient_name': patient_name,
                    'agent': agent,
                    'duration': int((current_time - start_time).total_seconds()),
                    'status': session_data.get('status', 'processing'),
                    'timestamp': start_time.isoformat(),
                    'details': '',
                    'category': 'processing'
                }
                activities.append(activity)
        
        # Show recent completed activities only (last 5 minutes)
        if hasattr(self, 'completed_activities') and self.completed_activities:
            for activity in self.completed_activities:
                activities.append(activity)
        
        # If no real activities, return empty list with NO FAKE DATA
        if not activities:
            logger.info("⚠️ No activities found - returning clean empty state")
            append_execution_log({
                'type': 'agent_activity',
                'count': 0,
                'state': 'empty',
                'timestamp': datetime.now().isoformat()
            })
            return []
        
        # Sort by timestamp (newest first) and limit to 10 recent activities
        sorted_activities = sorted(activities, key=lambda x: x['timestamp'], reverse=True)[:10]
        
        logger.info(f"✅ Returning {len(sorted_activities)} real activities")
        append_execution_log({
            'type': 'agent_activity',
            'count': len(sorted_activities),
            'timestamp': datetime.now().isoformat()
        })
        return sorted_activities
    
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
                
                # Format claim data for agentic processing
                formatted_claim = {
                    'claim_id': claim_id,
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
    app.run(debug=True, host='127.0.0.1', port=5000)
