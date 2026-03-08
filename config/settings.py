# config/settings.py

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # === AWS Core ===
    AWS_ACCESS_KEY_ID      = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY  = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_DEFAULT_REGION     = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    AWS_ACCOUNT_ID         = os.getenv("AWS_ACCOUNT_ID", "390783052961")

    # === Bedrock LLM (IAM key auth via boto3) ===
    AWS_BEDROCK_MODEL_ID = os.getenv("AWS_BEDROCK_MODEL_ID", "us.amazon.nova-micro-v1:0")

    # === S3 Storage ===
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "alpha-claims-demo-390783052961")

    # === Bedrock Agent Core — 6 Agents ===
    BEDROCK_AGENT_RISK_PREDICTOR   = os.getenv("BEDROCK_AGENT_RISK_PREDICTOR",   "XLAYW801JO")
    BEDROCK_AGENT_APPEAL_GENERATOR = os.getenv("BEDROCK_AGENT_APPEAL_GENERATOR", "S4YKZVC69F")
    BEDROCK_AGENT_AUTO_CORRECTOR   = os.getenv("BEDROCK_AGENT_AUTO_CORRECTOR",   "53KNQP1PMD")
    BEDROCK_AGENT_CLAIM_SUBMITTER  = os.getenv("BEDROCK_AGENT_CLAIM_SUBMITTER",  "BSI3CF17OU")
    BEDROCK_AGENT_RESUBMITTER      = os.getenv("BEDROCK_AGENT_RESUBMITTER",      "VCKGXKAZN0")
    BEDROCK_AGENT_FEEDBACK_LEARNER = os.getenv("BEDROCK_AGENT_FEEDBACK_LEARNER", "WUKCB3RG8Z")
    BEDROCK_AGENT_ALIAS            = os.getenv("BEDROCK_AGENT_ALIAS",            "TSTALIASID")
    BEDROCK_AGENTS_ROLE_ARN        = os.getenv("BEDROCK_AGENTS_ROLE_ARN", "arn:aws:iam::390783052961:role/BedrockAgentsClaimsRole")

    # === System ===
    OPERATIONAL_MODE = os.getenv("OPERATIONAL_MODE", "mcp")
    MCP_SERVER_URL   = os.getenv("MCP_SERVER_URL", "http://localhost:8001")
    DATA_SOURCE      = os.getenv("DATA_SOURCE", "openemr")

    # Agent settings
    RISK_THRESHOLD = 0.4
    LOG_LEVEL      = "INFO"
    TIMEOUT        = 15

    # Compliance
    ENABLE_LOG_REDACTION = True
    REDACTED_FIELDS      = ["patient_name", "dob", "insurance_id"]

    # Insurer API routing
    PRIMARY_API_URL   = "http://localhost:8081"   # BlueCross/Aetna
    SECONDARY_API_URL = "http://localhost:8082"   # Cigna/United
    DUMMY_API_URL     = "http://localhost:8081"

    @staticmethod
    def validate():
        missing = []
        if not Settings.AWS_ACCESS_KEY_ID:
            missing.append("AWS_ACCESS_KEY_ID")
        if not Settings.AWS_SECRET_ACCESS_KEY:
            missing.append("AWS_SECRET_ACCESS_KEY")
        if missing:
            raise EnvironmentError(f"Missing required env variables: {', '.join(missing)}")
        print(f"[OK] Bedrock LLM  : {Settings.AWS_BEDROCK_MODEL_ID} @ {Settings.AWS_DEFAULT_REGION}")
        print(f"[OK] S3 Bucket    : {Settings.S3_BUCKET_NAME}")
        print(f"[OK] Bedrock Agents: RiskPredictor={Settings.BEDROCK_AGENT_RISK_PREDICTOR} | AppealGen={Settings.BEDROCK_AGENT_APPEAL_GENERATOR}")

Settings.validate()
