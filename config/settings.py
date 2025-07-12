# config/settings.py

import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class Settings:
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    # System operational mode - MCP for full agentic behavior
    OPERATIONAL_MODE = os.getenv("OPERATIONAL_MODE", "mcp")  # Default to MCP mode
    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001")

    # LangGraph / Agent settings
    RISK_THRESHOLD = 0.4
    LOG_LEVEL = "INFO"
    TIMEOUT = 15  # seconds for external API calls

    # Compliance
    ENABLE_LOG_REDACTION = True
    REDACTED_FIELDS = ["patient_name", "dob", "insurance_id"]

    # API configurations for multiple insurers
    PRIMARY_API_URL = "http://localhost:8081"  # BlueCross/Aetna
    SECONDARY_API_URL = "http://localhost:8082"  # Cigna/United
    
    # Legacy support
    DUMMY_API_URL = "http://localhost:8081"

    @staticmethod
    def validate():
        missing = [
            key for key in [
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_VERSION",
                "AZURE_OPENAI_DEPLOYMENT_NAME"
            ]
            if not getattr(Settings, key)
        ]
        if missing:
            raise EnvironmentError(f"Missing required env variables: {missing}")

# Run validation on import
Settings.validate()
