"""
Centralized Execution Logger for Healthcare Claims Processing System
Captures all execution details across all components
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, List, Optional

class ExecutionLogger:
    """Centralized logger for all system execution details"""
    
    def __init__(self, log_dir: str = None):
        self.log_dir = Path(log_dir) if log_dir else Path(__file__).parent.parent / 'data' / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup different log files for different components
        self.loggers = {}
        self._setup_loggers()
        
    def _setup_loggers(self):
        """Setup rotating file loggers for different components"""
        components = [
            'api_server', 'claim_flow', 'risk_predictor', 'auto_corrector',
            'claim_submitter', 'appeal_generator', 'resubmitter', 'feedback_learner',
            'mcp_client', 'insurer_api', 'execution_trace'
        ]
        
        for component in components:
            logger = logging.getLogger(f"healthcare.{component}")
            logger.setLevel(logging.INFO)
            
            # Remove existing handlers to avoid duplicates
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Create rotating file handler
            log_file = self.log_dir / f"{component}.log"
            handler = RotatingFileHandler(
                log_file, 
                maxBytes=5_000_000,  # 5MB
                backupCount=5,
                encoding='utf-8'
            )
            
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            self.loggers[component] = logger
    
    def log_execution_step(self, component: str, step: str, data: Dict[str, Any] = None, level: str = "INFO"):
        """Log a detailed execution step"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Clean data for JSON serialization
            clean_data = self._clean_for_json(data) if data else {}
            
            # Create detailed log entry
            log_entry = {
                'timestamp': timestamp,
                'component': component,
                'step': step,
                'data': clean_data,
                'level': level
            }
            
            # Log to component-specific file
            if component in self.loggers:
                message = f"{step} | Data: {json.dumps(clean_data, indent=None)}"
                if level.upper() == "ERROR":
                    self.loggers[component].error(message)
                elif level.upper() == "WARNING":
                    self.loggers[component].warning(message)
                else:
                    self.loggers[component].info(message)
            
            # Also log to unified execution trace
            self._write_jsonl(self.log_dir / 'execution_trace.jsonl', log_entry)
            
        except Exception as e:
            print(f"[ExecutionLogger] Error logging: {e}")
    
    def log_claim_start(self, claim_data: Dict[str, Any]):
        """Log the start of claim processing"""
        self.log_execution_step(
            'claim_flow', 
            'CLAIM_PROCESSING_START',
            {
                'claim_id': claim_data.get('claim_id', 'unknown'),
                'patient_id': claim_data.get('patient_id', 'unknown'),
                'patient_name': claim_data.get('patient_name', 'unknown'),
                'procedure_code': claim_data.get('procedure_code', 'unknown'),
                'claim_amount': claim_data.get('claim_amount', 0),
                'insurer': claim_data.get('insurer', 'unknown')
            }
        )
    
    def log_agent_processing(self, agent_name: str, action: str, input_data: Dict[str, Any], output_data: Dict[str, Any] = None):
        """Log agent processing details"""
        self.log_execution_step(
            agent_name.lower().replace(' ', '_'),
            f"AGENT_{action.upper()}",
            {
                'agent': agent_name,
                'action': action,
                'input': input_data,
                'output': output_data or {}
            }
        )
    
    def log_api_call(self, api_name: str, endpoint: str, request_data: Dict[str, Any], response_data: Dict[str, Any], status_code: int = 200):
        """Log API calls and responses"""
        self.log_execution_step(
            'api_calls',
            f"API_{api_name.upper()}_{endpoint.upper()}",
            {
                'api': api_name,
                'endpoint': endpoint,
                'request': request_data,
                'response': response_data,
                'status_code': status_code
            }
        )
    
    def log_error(self, component: str, error: str, context: Dict[str, Any] = None):
        """Log errors with context"""
        self.log_execution_step(
            component,
            'ERROR',
            {
                'error_message': str(error),
                'context': context or {}
            },
            level="ERROR"
        )
    
    def log_workflow_step(self, step_name: str, claim_id: str, step_data: Dict[str, Any]):
        """Log workflow progression"""
        self.log_execution_step(
            'workflow',
            f"WORKFLOW_{step_name.upper()}",
            {
                'claim_id': claim_id,
                'step': step_name,
                'step_data': step_data
            }
        )
    
    def _clean_for_json(self, obj):
        """Clean object for JSON serialization"""
        if obj is None:
            return None
        elif isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, str):
            # Clean problematic Unicode characters
            cleaned = obj.replace('\u2013', '-').replace('\u2014', '--')
            cleaned = cleaned.replace('\u2018', "'").replace('\u2019', "'")
            cleaned = cleaned.replace('\u201C', '"').replace('\u201D', '"')
            return ''.join(char if ord(char) < 128 else '?' for char in cleaned)
        elif hasattr(obj, '__dict__'):
            return self._clean_for_json(obj.__dict__)
        else:
            return str(obj)
    
    def _write_jsonl(self, file_path: Path, data: Dict[str, Any]):
        """Write data to JSONL file"""
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=True) + '\n')
        except Exception as e:
            print(f"[ExecutionLogger] Error writing to {file_path}: {e}")
    
    def get_recent_logs(self, component: str, lines: int = 50) -> List[str]:
        """Get recent log lines for a component"""
        try:
            log_file = self.log_dir / f"{component}.log"
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    return f.readlines()[-lines:]
            return []
        except Exception as e:
            print(f"[ExecutionLogger] Error reading logs for {component}: {e}")
            return []

# Global logger instance
execution_logger = ExecutionLogger()

def log_execution(component: str, step: str, data: Dict[str, Any] = None, level: str = "INFO"):
    """Convenience function for logging execution steps"""
    execution_logger.log_execution_step(component, step, data, level)

def log_agent_work(agent_name: str, action: str, input_data: Dict[str, Any], output_data: Dict[str, Any] = None):
    """Convenience function for logging agent work"""
    execution_logger.log_agent_processing(agent_name, action, input_data, output_data)

def log_error(component: str, error: str, context: Dict[str, Any] = None):
    """Convenience function for logging errors"""
    execution_logger.log_error(component, error, context)
