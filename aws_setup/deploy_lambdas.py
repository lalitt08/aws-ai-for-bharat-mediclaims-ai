"""
deploy_lambdas.py
=================
Deploys all 6 Lambda Action Group handlers and attaches them to Bedrock Agents.

Run from project root:
    python aws_setup/deploy_lambdas.py

What it does:
  1. Zips each Lambda handler + shared utils
  2. Creates/updates Lambda functions in us-east-1
  3. Adds resource-based policy so Bedrock can invoke each Lambda
  4. Creates/updates Action Groups on each Bedrock Agent
  5. Prepares each agent and creates a live alias
  6. Prints updated alias IDs to paste into .env
"""

import boto3
import json
import os
import sys
import zipfile
import io
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
REGION      = "us-east-1"
ACCOUNT_ID  = "390783052961"
ROLE_ARN    = f"arn:aws:iam::{ACCOUNT_ID}:role/BedrockAgentsClaimsRole"
S3_BUCKET   = "alpha-claims-demo-390783052961"
RUNTIME     = "python3.11"

# Lambda env vars injected into every function — IAM key auth, Nova Micro
LAMBDA_ENV = {
    "S3_BUCKET_NAME":        S3_BUCKET,
    "APP_REGION":            REGION,
    "AWS_BEDROCK_MODEL_ID":  "us.amazon.nova-micro-v1:0",
    "PRIMARY_INSURER_API":   "http://localhost:8081",
    "SECONDARY_INSURER_API": "http://localhost:8082",
}

# Agent definitions: key → (agent_id, lambda_name, handler_dir, action_group_name)
AGENTS = {
    "risk_predictor": {
        "agent_id":    "XLAYW801JO",
        "lambda_name": "mediclaims-risk-predictor",
        "handler_dir": "lambda/risk_predictor",
        "ag_name":     "RiskPredictorActions",
        "description": "Validates ICD/CPT codes, checks prior auth, analyzes denial patterns",
    },
    "auto_corrector": {
        "agent_id":    "53KNQP1PMD",
        "lambda_name": "mediclaims-auto-corrector",
        "handler_dir": "lambda/auto_corrector",
        "ag_name":     "AutoCorrectorActions",
        "description": "Corrects ICD/CPT codes, generates prior auth, validates NPI",
    },
    "claim_submitter": {
        "agent_id":    "BSI3CF17OU",
        "lambda_name": "mediclaims-claim-submitter",
        "handler_dir": "lambda/claim_submitter",
        "ag_name":     "ClaimSubmitterActions",
        "description": "Checks eligibility, submits claims to insurer APIs, saves results",
    },
    "appeal_generator": {
        "agent_id":    "S4YKZVC69F",
        "lambda_name": "mediclaims-appeal-generator",
        "handler_dir": "lambda/appeal_generator",
        "ag_name":     "AppealGeneratorActions",
        "description": "Retrieves denial details, generates appeal letters, saves to S3",
    },
    "resubmitter": {
        "agent_id":    "VCKGXKAZN0",
        "lambda_name": "mediclaims-resubmitter",
        "handler_dir": "lambda/resubmitter",
        "ag_name":     "ResubmitterActions",
        "description": "Determines resubmission strategy, resubmits with appeal, updates status",
    },
    "feedback_learner": {
        "agent_id":    "WUKCB3RG8Z",
        "lambda_name": "mediclaims-feedback-learner",
        "handler_dir": "lambda/feedback_learner",
        "ag_name":     "FeedbackLearnerActions",
        "description": "Records outcomes, updates denial patterns, provides learning insights",
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def project_root() -> str:
    """Return the alpha (7)/alpha directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)  # one level up from aws_setup/


def build_zip(handler_dir: str) -> bytes:
    """Zip handler.py + shared/utils.py into a bytes buffer."""
    root = project_root()
    buf  = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # handler.py → handler.py (root of zip)
        handler_path = os.path.join(root, handler_dir, "handler.py")
        zf.write(handler_path, "handler.py")
        # shared utils → shared/utils.py
        shared_path = os.path.join(root, "lambda", "shared", "utils.py")
        zf.write(shared_path, "shared/__init__.py".replace("__init__", "utils"))
        # empty __init__ so Python treats shared/ as a package
        zf.writestr("shared/__init__.py", "")
    buf.seek(0)
    return buf.read()


def lambda_client():
    return boto3.client("lambda", region_name=REGION)


def bedrock_client():
    return boto3.client("bedrock-agent", region_name=REGION)


def create_or_update_lambda(name: str, zip_bytes: bytes, description: str) -> str:
    """Create or update a Lambda function. Returns the function ARN."""
    lc = lambda_client()
    try:
        resp = lc.get_function(FunctionName=name)
        # Wait for any in-progress update to finish
        for _ in range(10):
            state = lc.get_function_configuration(FunctionName=name)
            if state.get("LastUpdateStatus") in (None, "Successful"):
                break
            log.info(f"  Waiting for Lambda {name} to be ready...")
            time.sleep(5)
        # Update existing
        lc.update_function_code(FunctionName=name, ZipFile=zip_bytes)
        time.sleep(5)  # wait for code update
        lc.update_function_configuration(
            FunctionName=name,
            Environment={"Variables": LAMBDA_ENV},
            Timeout=30,
            MemorySize=256,
        )
        arn = resp["Configuration"]["FunctionArn"]
        log.info(f"  Updated Lambda: {name}")
    except lc.exceptions.ResourceNotFoundException:
        resp = lc.create_function(
            FunctionName=name,
            Runtime=RUNTIME,
            Role=ROLE_ARN,
            Handler="handler.lambda_handler",
            Code={"ZipFile": zip_bytes},
            Description=description,
            Timeout=30,
            MemorySize=256,
            Environment={"Variables": LAMBDA_ENV},
        )
        arn = resp["FunctionArn"]
        log.info(f"  Created Lambda: {name} → {arn}")
        # Wait for Active state
        for _ in range(15):
            state = lc.get_function_configuration(FunctionName=name)
            if state.get("State") == "Active":
                break
            log.info(f"  Waiting for {name} to become Active...")
            time.sleep(4)
    return arn


def add_bedrock_permission(lambda_name: str, agent_id: str):
    """Allow Bedrock Agent to invoke this Lambda."""
    lc = lambda_client()
    sid = f"AllowBedrock-{agent_id}"
    try:
        lc.remove_permission(FunctionName=lambda_name, StatementId=sid)
    except Exception:
        pass
    try:
        lc.add_permission(
            FunctionName=lambda_name,
            StatementId=sid,
            Action="lambda:InvokeFunction",
            Principal="bedrock.amazonaws.com",
            SourceArn=f"arn:aws:bedrock:{REGION}:{ACCOUNT_ID}:agent/{agent_id}",
        )
        log.info(f"  Permission added: Bedrock agent {agent_id} → {lambda_name}")
    except Exception as e:
        log.warning(f"  Permission add failed (may already exist): {e}")


def load_openapi_schema(handler_dir: str) -> str:
    root = project_root()
    schema_path = os.path.join(root, handler_dir, "openapi_schema.json")
    with open(schema_path, "r") as f:
        return f.read()


def attach_action_group(agent_id: str, ag_name: str, lambda_arn: str,
                         openapi_schema: str, description: str):
    """Create or update an Action Group on a Bedrock Agent."""
    bc = bedrock_client()

    # List existing action groups
    existing_ags = bc.list_agent_action_groups(
        agentId=agent_id, agentVersion="DRAFT"
    ).get("actionGroupSummaries", [])

    existing = next((ag for ag in existing_ags if ag["actionGroupName"] == ag_name), None)

    ag_kwargs = dict(
        agentId=agent_id,
        agentVersion="DRAFT",
        actionGroupName=ag_name,
        description=description,
        actionGroupExecutor={"lambda": lambda_arn},
        apiSchema={"payload": openapi_schema},
        actionGroupState="ENABLED",
    )

    if existing:
        bc.update_agent_action_group(
            actionGroupId=existing["actionGroupId"],
            **ag_kwargs,
        )
        log.info(f"  Updated Action Group: {ag_name} on agent {agent_id}")
    else:
        bc.create_agent_action_group(**ag_kwargs)
        log.info(f"  Created Action Group: {ag_name} on agent {agent_id}")


def prepare_agent_and_alias(agent_id: str, agent_name: str) -> str:
    """
    Prepare agent DRAFT, then create/update a 'live' alias.
    Creating an alias WITHOUT routingConfiguration auto-creates a new version
    from DRAFT and points the alias to it — this is the correct AWS pattern.
    Returns alias ID.
    """
    bc = bedrock_client()

    log.info(f"  Preparing agent {agent_id}...")
    bc.prepare_agent(agentId=agent_id)

    # Wait for PREPARED
    for _ in range(20):
        time.sleep(5)
        try:
            status = bc.get_agent(agentId=agent_id)["agent"].get("agentStatus", "")
            log.info(f"  Agent {agent_id} status: {status}")
            if status == "PREPARED":
                break
        except Exception as e:
            log.warning(f"  Status check failed: {e}")

    # Find existing 'live' alias
    aliases = bc.list_agent_aliases(agentId=agent_id).get("agentAliasSummaries", [])
    live_alias = next((a for a in aliases if a["agentAliasName"] == "live"), None)

    if live_alias:
        alias_id = live_alias["agentAliasId"]
        # Delete and recreate so a new version is snapshotted from DRAFT
        try:
            bc.delete_agent_alias(agentId=agent_id, agentAliasId=alias_id)
            time.sleep(2)
            log.info(f"  Deleted old alias {alias_id}")
        except Exception as e:
            log.warning(f"  Could not delete old alias: {e}")

    # Create alias WITHOUT routingConfiguration → Bedrock auto-creates new version from DRAFT
    try:
        resp = bc.create_agent_alias(agentId=agent_id, agentAliasName="live")
        alias_id = resp["agentAlias"]["agentAliasId"]
        routing  = resp["agentAlias"].get("routingConfiguration", [])
        ver = routing[0].get("agentVersion", "?") if routing else "?"
        log.info(f"  Updated alias 'live': {alias_id} → version {ver}")
    except Exception as e:
        log.warning(f"  Alias creation failed: {e}")
        alias_id = "TSTALIASID"

    return alias_id


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("MediClaims — Lambda Action Groups Deployment")
    log.info("=" * 60)

    alias_map = {}

    for key, cfg in AGENTS.items():
        log.info(f"\n[{key.upper()}]")

        # 1. Build zip
        log.info("  Building zip...")
        zip_bytes = build_zip(cfg["handler_dir"])
        log.info(f"  Zip size: {len(zip_bytes):,} bytes")

        # 2. Deploy Lambda
        lambda_arn = create_or_update_lambda(
            cfg["lambda_name"], zip_bytes, cfg["description"]
        )

        # 3. Grant Bedrock permission
        add_bedrock_permission(cfg["lambda_name"], cfg["agent_id"])

        # 4. Load OpenAPI schema
        schema = load_openapi_schema(cfg["handler_dir"])

        # 5. Attach Action Group
        attach_action_group(
            cfg["agent_id"], cfg["ag_name"], lambda_arn, schema, cfg["description"]
        )

        # 6. Prepare agent + create alias
        alias_id = prepare_agent_and_alias(cfg["agent_id"], key)
        alias_map[key] = alias_id

    # ── Print .env updates ────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("DEPLOYMENT COMPLETE — Add these to your .env:")
    log.info("=" * 60)
    env_lines = []
    for key, alias_id in alias_map.items():
        env_key = f"BEDROCK_AGENT_ALIAS_{key.upper()}"
        line = f"{env_key}={alias_id}"
        print(line)
        env_lines.append(line)

    # Write alias map to a file for reference
    root = project_root()
    alias_file = os.path.join(root, "aws_setup", "lambda_aliases.json")
    with open(alias_file, "w") as f:
        json.dump(alias_map, f, indent=2)
    log.info(f"\nAlias map saved to: {alias_file}")
    log.info("\nAll 6 agents now have real Lambda Action Groups attached.")
    log.info("Agents will use tools instead of stalling on missing functions.")


if __name__ == "__main__":
    main()
