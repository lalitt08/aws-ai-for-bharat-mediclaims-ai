"""
Configuration settings for Post-Submission Appeals Dashboard
"""

import os
from typing import Dict, Any

class Settings:
    """Application settings and configuration"""
    
    # Application settings
    APP_NAME = "MediClaims AI - Post-Submission Appeals Dashboard"
    VERSION = "1.0.0"
    DEBUG = True
    
    # Server settings
    HOST = "0.0.0.0"
    PORT = 8000
    
    # Database settings (using SQLite for demo)
    DATABASE_URL = "sqlite:///./appeals_demo.db"
    DATABASE_FILE = "appeals_demo.db"
    
    # Mock data settings
    MOCK_DATA_ENABLED = True
    GENERATE_DEMO_DATA = True
    NUM_MOCK_APPEALS = 50
    NUM_MOCK_DENIALS = 25
    
    # Appeal status categories
    APPEAL_STATUSES = {
        "pending": "Pending Review",
        "active": "In Progress", 
        "denied": "Denied",
        "approved": "Approved",
        "resubmitted": "Resubmitted",
        "closed": "Closed"
    }
    
    # Denial reason categories
    DENIAL_CATEGORIES = {
        "medical_necessity": "Medical Necessity",
        "prior_authorization": "Prior Authorization Required",
        "documentation": "Insufficient Documentation",
        "coding_error": "Coding Error",
        "policy_exclusion": "Policy Exclusion",
        "timely_filing": "Timely Filing Limit",
        "duplicate_claim": "Duplicate Claim",
        "coordination_benefits": "Coordination of Benefits"
    }
    
    # Compliance rules
    COMPLIANCE_RULES = {
        "hipaa_required": True,
        "state_regulations": True,
        "payer_policies": True,
        "timely_filing_days": 365,
        "appeal_deadline_days": 60
    }
    
    # Mock payer APIs
    MOCK_PAYERS = {
        "aetna": {
            "name": "Aetna",
            "api_url": "https://api.aetna.com/v1/claims",
            "success_rate": 0.75
        },
        "united": {
            "name": "United Healthcare",
            "api_url": "https://api.united.com/v1/claims", 
            "success_rate": 0.68
        },
        "bluecross": {
            "name": "BlueCross BlueShield",
            "api_url": "https://api.bcbs.com/v1/claims",
            "success_rate": 0.72
        },
        "cigna": {
            "name": "Cigna",
            "api_url": "https://api.cigna.com/v1/claims",
            "success_rate": 0.70
        }
    }
    
    # ERA/835 processing settings
    ERA_SETTINGS = {
        "auto_process": True,
        "batch_size": 100,
        "retry_attempts": 3,
        "supported_formats": ["835", "ERA", "XML", "JSON"]
    }
    
    # AI/ML settings (mock)
    AI_SETTINGS = {
        "denial_prediction_threshold": 0.7,
        "appeal_success_threshold": 0.6,
        "auto_correction_enabled": True,
        "compliance_validation": True
    }
    
    # UI settings
    UI_SETTINGS = {
        "theme": "healthcare",
        "items_per_page": 20,
        "auto_refresh_interval": 30,  # seconds
        "show_advanced_features": True
    }
    
    @classmethod
    def get_database_path(cls) -> str:
        """Get the full path to the database file"""
        return os.path.join(os.getcwd(), cls.DATABASE_FILE)
    
    @classmethod
    def get_mock_data_config(cls) -> Dict[str, Any]:
        """Get mock data generation configuration"""
        return {
            "enabled": cls.MOCK_DATA_ENABLED,
            "appeals_count": cls.NUM_MOCK_APPEALS,
            "denials_count": cls.NUM_MOCK_DENIALS,
            "generate_fresh": cls.GENERATE_DEMO_DATA
        }
    
    @classmethod
    def get_payer_config(cls, payer_id: str) -> Dict[str, Any]:
        """Get configuration for a specific payer"""
        return cls.MOCK_PAYERS.get(payer_id, {})
    
    @classmethod
    def get_all_payers(cls) -> Dict[str, Dict[str, Any]]:
        """Get all payer configurations"""
        return cls.MOCK_PAYERS

# Global settings instance
settings = Settings()
