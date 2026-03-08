# Healthcare Claims AI — AWS Migration Progress

## AWS Account
- Account ID: 390783052961
- IAM User: admin (AdministratorAccess)
- Region: us-east-1
- Credentials: AKIAVV7D4FSQ2JNOUB5J (static IAM key, configured in ~/.aws/credentials)

---

## Architecture Overview

```
Pre-Submission (Flask :5000)          Post-Submission (FastAPI :8003)
        |                                       |
   LangGraph                              FastAPI routers
   6 AI Agents                            ERA processor
        |                                       |
        +------------- AWS LAYER ---------------+
                            |
              ┌─────────────┼─────────────┐
              |             |             |
         S3 Bucket    Bedrock LLM    Bedrock Agents
     (claim data,    (Llama 3 70B)   (6 agents hosted
      ERA files,                      on Agent Core)
      appeals,
      patients CSV)
```

---

## TASK 1: Understand project structure — DONE
- Pre-Submission: LangGraph multi-agent (6 agents), Flask :5000, MCP :8001, insurers :8081/:8082
- Post-Submission: FastAPI :8003, ERA upload, appeals dashboard

## TASK 2: Connect both systems — DONE
- Shared claim_status.json + patients1.csv
- Unified launcher: start_all.py, start_all.bat

## TASK 3: Real data + UI unification — DONE
- All 4 API routers use real data
- Unified design system, Pre/Post navigation buttons

## TASK 4: ERA upload + LLM analysis — DONE
- era_processor.py parses real X12 835 segments
- LLM via AWS Bedrock bearer token (Llama 3 70B)
- Patient-specific ERA sample files

## TASK 5: Migrate to AWS Bedrock — DONE
- BedrockLLM wrapper: tools/bedrock_llm.py
- All agents use Bedrock (appeal_generator, risk_predictor, era_processor)
- Model: meta.llama3-70b-instruct-v1:0 @ us-west-2 via bearer token

## TASK 6: Claim Journey on post-submission profile — DONE
- /api/claim-journey/{patient_id} endpoint
- Timeline UI in patient-details.html

## TASK 7: Claim details modal in pre-submission — DONE
- viewClaimDetails() modal in dashboard.js
- Shows claim fields, denial reason, "View in Post-Submission" link

## TASK 8: Real X12 837P for all patients — DONE
- tools/x12_837p_builder.py — full ANSI X12 837P
- api_server.py submit_claim() now enriches with full CSV fields (dob, gender, address, npi)
- /api/claim-x12/<patient_id> endpoint
- Modal shows collapsible X12 block in dark terminal style

---

## TASK 10: Deploy to AWS — DONE ✅

### Live URLs (paste in browser)
- Pre-Submission Dashboard:  http://mediclaims-alb-640335154.us-east-1.elb.amazonaws.com/
- Post-Submission Appeals:   http://mediclaims-alb-640335154.us-east-1.elb.amazonaws.com/appeals/
- Direct EC2 (backup):       http://13.220.20.244:5000/  (pre-submission)
                             http://13.220.20.244:8003/  (post-submission)

### AWS Infrastructure
- EC2: i-003e68fd0c2d31cf3 (t3.medium, Amazon Linux 2, us-east-1a)
  - Public IP: 13.220.20.244
  - IAM Role: MediClaimsEC2Role (S3 + Bedrock access)
  - Python: 3.8 (via amazon-linux-extras)
  - Packages: flask, fastapi, uvicorn, boto3, pandas, etc. installed for python3.8
  - Runs: Flask :5000, FastAPI :8003, nginx :80

- ALB: mediclaims-alb-640335154.us-east-1.elb.amazonaws.com
  - Port 80 → /appeals/* → Post-submission FastAPI (port 8003)
  - Port 80 → default → Pre-submission Flask (port 5000)
  - Both targets: HEALTHY

### To redeploy after code changes:
  1. Compress-Archive -Path "alpha (7)/alpha/*" -DestinationPath "mediclaims_new.zip" -Force
     (or use the PowerShell zip script that excludes .git/__pycache__/.env)
  2. aws s3 cp mediclaims_new.zip s3://alpha-claims-demo-390783052961/app/mediclaims.zip
  3. Run SSM redeploy.json command on i-003e68fd0c2d31cf3

### AWS Infrastructure
- EC2: i-003e68fd0c2d31cf3 (t3.medium, Amazon Linux 2, us-east-1a)
  - Public IP: 13.220.20.244
  - IAM Role: MediClaimsEC2Role (S3 + Bedrock access)
  - 20GB gp3 volume
  - Runs: Flask :5000, FastAPI :8003, MCP :8001, Insurer APIs :8081/:8082, nginx :80

- ALB: mediclaims-alb-640335154.us-east-1.elb.amazonaws.com
  - Port 80 → / → Pre-submission Flask (port 5000)
  - Port 80 → /appeals/* → Post-submission FastAPI (port 8003)
  - Security Group: sg-030b71d923a5c1212

- S3: alpha-claims-demo-390783052961
  - app/mediclaims.zip (full app code)
  - patients/patients1.csv
  - claims/claim_status.json
  - era/*.835 files
  - appeals/ (auto-populated)

- Bedrock Agents (6 agents, all PREPARED):
  RiskPredictorAgent    XLAYW801JO
  AppealGeneratorAgent  S4YKZVC69F
  AutoCorrectorAgent    53KNQP1PMD
  ClaimSubmitterAgent   BSI3CF17OU
  ResubmitterAgent      VCKGXKAZN0
  FeedbackLearnerAgent  WUKCB3RG8Z

### To redeploy after code changes:
  1. Compress-Archive -Path "alpha (7)/alpha/*" -DestinationPath "mediclaims.zip" -Force
  2. aws s3 cp mediclaims.zip s3://alpha-claims-demo-390783052961/app/mediclaims.zip
  3. SSH to 13.220.20.244 and run: cd /opt/mediclaims && aws s3 cp s3://alpha-claims-demo-390783052961/app/mediclaims.zip . && unzip -o mediclaims.zip -d . && systemctl restart mediclaims-pre mediclaims-post


### AWS Resources Created (Account: 390783052961, Region: us-east-1)

S3 Bucket: alpha-claims-demo-390783052961
  - patients/patients1.csv       (6032 bytes — all 25 patients)
  - claims/claim_status.json     (8685 bytes — 12 processed claims)
  - claims/denial_patterns.json  (2308 bytes)
  - era/PAT002_Sarah_Johnson.835
  - era/PAT004_Emma_Wilson.835
  - era/PAT011_Kevin_Anderson.835
  - era/PAT015_Andrew_Harris.835
  - era/PAT017_Matthew_Lewis.835
  - era/test_simple.835
  - appeals/   (auto-populated when appeals are generated)
  - logs/      (auto-populated with feedback learner JSONL)

IAM Role: BedrockAgentsClaimsRole
  ARN: arn:aws:iam::390783052961:role/BedrockAgentsClaimsRole
  Policy: Bedrock InvokeModel + S3 read/write on bucket

Bedrock Agents (all PREPARED, alias TSTALIASID, model: meta.llama3-70b-instruct-v1:0):
  RiskPredictorAgent    XLAYW801JO
  AppealGeneratorAgent  S4YKZVC69F
  AutoCorrectorAgent    53KNQP1PMD
  ClaimSubmitterAgent   BSI3CF17OU
  ResubmitterAgent      VCKGXKAZN0
  FeedbackLearnerAgent  WUKCB3RG8Z

### New Files Created
- tools/s3_storage.py              — S3 read/write for all data (claims, ERA, appeals, X12, logs)
- tools/bedrock_agent_client.py    — Invokes Bedrock Agent Core for each agent step
- aws_setup/trust_policy.json      — IAM trust policy for Bedrock
- aws_setup/bedrock_agent_policy.json — IAM inline policy (Bedrock + S3)
- aws_setup/test_aws.py            — Connectivity test (run to verify)

### Updated Files
- config/settings.py               — Added S3_BUCKET_NAME, all 6 BEDROCK_AGENT_* IDs
- .env                             — Added AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME, all agent IDs
- tools/claim_status_manager.py    — S3 primary storage, local fallback; auto-uploads X12 to S3
- agents/appeal_generator.py       — Calls Bedrock Agent Core first, uploads PDF to S3
- agents/feedback_learner.py       — Calls Bedrock Agent Core, saves insights to S3 logs
- post_submission_demo/services/era_processor.py — Uploads ERA files to S3 after processing
- requirements.txt                 — Replaced azure-* with boto3>=1.28.0

### Data Flow (AWS)
  Claim submitted → LangGraph pipeline runs locally
    → Each agent optionally calls its Bedrock Agent Core for enhanced AI
    → claim_status.json saved to S3 (+ local fallback)
    → X12 837P uploaded to S3 (claims/x12/)
    → Appeal PDFs uploaded to S3 (appeals/)
    → ERA files uploaded to S3 (era/)
    → Feedback insights logged to S3 (logs/feedback_learner.jsonl)


---

## Key File Paths
- Pre-submission entry: alpha (7)/alpha/web_dashboard/api_server.py
- Post-submission entry: alpha (7)/alpha/post_submission_demo/app.py
- LangGraph flow: alpha (7)/alpha/graph/claim_flow.py
- Agent nodes: alpha (7)/alpha/graph/nodes.py
- All agents: alpha (7)/alpha/agents/
- Bedrock LLM: alpha (7)/alpha/tools/bedrock_llm.py
- X12 builder: alpha (7)/alpha/tools/x12_837p_builder.py
- S3 storage (NEW): alpha (7)/alpha/tools/s3_storage.py
- Bedrock agent client (NEW): alpha (7)/alpha/tools/bedrock_agent_client.py
- Settings: alpha (7)/alpha/config/settings.py
- .env: alpha (7)/alpha/.env

## Run Commands
- Pre-submission dashboard: cd "alpha (7)/alpha" && python web_dashboard/api_server.py
- Post-submission dashboard: cd "alpha (7)/alpha/post_submission_demo" && python app.py
- MCP server: cd "alpha (7)/alpha" && python mcp_server/main.py
- Insurer API primary: cd "alpha (7)/alpha" && python tools/insurer_api_primary.py
- Insurer API secondary: cd "alpha (7)/alpha" && python tools/insurer_api_secondary.py
- All at once: cd "alpha (7)/alpha" && python start_all.py

---

## TASK 11: Bedrock Agent Core with Lambda Action Groups — DONE ✅

### Architecture
- 6 Bedrock Agents (us-east-1), all using `us.amazon.nova-micro-v1:0` (IAM key auth, no marketplace needed)
- All LLM calls via boto3 SigV4 (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY) — no bearer tokens
- Each agent has a Lambda Action Group with real tool implementations
- All aliases point to version 3 (Nova Micro + Action Groups)

### Agent Aliases (v3 — live)
```
RiskPredictor:    XLAYW801JO → alias TLUEP83SGK → v3 (Nova Micro, 1 AG)
AutoCorrector:    53KNQP1PMD → alias TF1NCDOXCF → v3 (Nova Micro, 1 AG)
ClaimSubmitter:   BSI3CF17OU → alias N8WN159M4J → v3 (Nova Micro, 1 AG)
AppealGenerator:  S4YKZVC69F → alias MH6PV48HOH → v3 (Nova Micro, 1 AG)
Resubmitter:      VCKGXKAZN0 → alias UOOVAJIKTV → v3 (Nova Micro, 1 AG)
FeedbackLearner:  WUKCB3RG8Z → alias CXGYGDLI7A → v3 (Nova Micro, 1 AG)
```

### Test Results: 6/6 PASS
- RiskPredictor: validates ICD-10/CPT, checks prior auth, analyzes denial patterns
- AutoCorrector: corrects codes, validates NPI, generates prior auth
- ClaimSubmitter: checks eligibility, submits claims, saves results
- AppealGenerator: gets denial details, checks requirements, generates appeal letters
- Resubmitter: determines strategy, resubmits with appeal, updates status
- FeedbackLearner: records outcomes, updates denial patterns, returns insights

### Key Files
- `tools/bedrock_llm.py` — boto3-based LLM wrapper (IAM key auth, Nova Micro)
- `tools/bedrock_agent_client.py` — invokes Bedrock Agent Core (v3 aliases)
- `tools/bedrock_agent_integration.py` — structured wrappers for LangGraph agents
- `lambda/shared/utils.py` — shared utilities, `agent_response(body, http_method)`
- `lambda/*/handler.py` — 6 Lambda handlers with correct httpMethod passthrough
- `aws_setup/deploy_lambdas.py` — deploys all 6 Lambdas + updates Action Groups
- `aws_setup/test_agents.py` — end-to-end test (6/6 PASS)
