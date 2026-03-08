# MediClaims AI

An agentic AI system that automates healthcare insurance claims processing — from submission through denial, appeal, and resubmission — using six specialized AI agents orchestrated by LangGraph and powered by AWS.

Built as a prototype for **AI for Bharat** under the Healthcare & Life Sciences domain.

---

## What It Does

Healthcare providers lose significant revenue every year due to claim denials caused by missing data, incorrect codes, or incomplete documentation. Staff spend hours manually fixing and resubmitting claims with low success rates.

MediClaims AI automates the entire claims lifecycle:

1. Fetches patient data in real time from the hospital's system
2. Predicts which claims are likely to be denied before submission
3. Auto-corrects issues in the claim data
4. Submits claims to the insurer in standard X12 837P format
5. If denied, generates an AI-written appeal letter
6. Resubmits the corrected claim with the appeal
7. Learns from every outcome to improve future predictions

---

## The Six AI Agents

| Agent | Purpose |
|-------|---------|
| **Risk Predictor** | Scores each claim for denial probability using patient history, insurer patterns, and medical codes |
| **Auto Corrector** | Fixes missing demographics, validates ICD-10/CPT codes, generates prior authorizations |
| **Claim Submitter** | Verifies eligibility and submits claims via insurer APIs in ANSI X12 837P format |
| **Appeal Generator** | Writes a professional appeal letter tailored to the specific denial reason using Amazon Nova Micro |
| **Resubmitter** | Packages the corrected claim with the appeal and resubmits to the insurer |
| **Feedback Learner** | Analyzes every outcome and updates denial patterns to improve future predictions |

### Agent Flow

```
Claim Input
    │
    ▼
Risk Predictor ──► High Risk? ──► Auto Corrector ──► Claim Submitter
                                                           │
                   Low Risk? ──────────────────────────────┘
                                                           │
                                              ┌────────────┴────────────┐
                                           Approved                  Denied
                                              │                         │
                                       Feedback Learner        Appeal Generator
                                                                        │
                                                                  Resubmitter
                                                                        │
                                                               Feedback Learner
```

---

## AWS Services Used

| Service | Purpose |
|---------|---------|
| **Amazon EC2** (t3.medium) | Hosts the full application — Flask :5000 (pre-submission), FastAPI :8003 (post-submission), MCP server :8001, nginx reverse proxy |
| **Application Load Balancer** | Routes `/appeals/*` to FastAPI and all other traffic to Flask. Provides a single public URL |
| **Amazon Bedrock — Agent Core** | Six individual Bedrock agents, one per AI role. Each agent has its own ID, alias, and action group |
| **Amazon Bedrock — Nova Micro** | Foundation model (`us.amazon.nova-micro-v1:0`) used by all six agents for reasoning and generation |
| **AWS Lambda** | Six Lambda functions act as action groups for the Bedrock agents — each implements the real tool logic (code validation, eligibility check, appeal writing, etc.) |
| **Amazon S3** | Central storage for patient CSV, claim status JSON, ERA/835 files, generated appeal PDFs, X12 837P transactions, feedback logs, and the deployment zip |
| **AWS IAM** | `MediClaimsEC2Role` gives EC2 access to S3 and Bedrock. `BedrockAgentsClaimsRole` allows agents to invoke Lambda and read/write S3 |

### Bedrock Agent IDs

```
RiskPredictor    → XLAYW801JO
AutoCorrector    → 53KNQP1PMD
ClaimSubmitter   → BSI3CF17OU
AppealGenerator  → S4YKZVC69F
Resubmitter      → VCKGXKAZN0
FeedbackLearner  → WUKCB3RG8Z
```

### S3 Bucket Structure

```
alpha-claims-demo-390783052961/
├── patients/patients1.csv       ← 25 patients from hospital system
├── claims/claim_status.json     ← live claim tracking
├── claims/x12/                  ← ANSI X12 837P per patient
├── era/                         ← ERA/835 remittance files from insurers
├── appeals/                     ← generated PDF appeal letters
├── logs/feedback_learner.jsonl  ← outcome learning data
└── app/mediclaims.zip           ← deployment artifact
```

---

## Live Deployment

| URL | Description |
|-----|-------------|
| `http://mediclaims-alb-640335154.us-east-1.elb.amazonaws.com/` | Pre-submission dashboard |
| `http://mediclaims-alb-640335154.us-east-1.elb.amazonaws.com/appeals/` | Post-submission appeals |
| `http://13.220.20.244:5000` | Direct EC2 — pre-submission |
| `http://13.220.20.244:8003` | Direct EC2 — post-submission |

---

## Project Structure

```
├── agents/                  # Six AI agent implementations
│   ├── risk_predictor.py
│   ├── auto_corrector.py
│   ├── claim_submitter.py
│   ├── appeal_generator.py
│   ├── resubmitter.py
│   └── feedback_learner.py
├── graph/                   # LangGraph workflow
│   ├── claim_flow.py        # Orchestration + branching logic
│   └── nodes.py             # Agent node wrappers
├── lambda/                  # Lambda action group handlers
│   ├── risk_predictor/
│   ├── auto_corrector/
│   ├── claim_submitter/
│   ├── appeal_generator/
│   ├── resubmitter/
│   └── feedback_learner/
├── web_dashboard/           # Pre-submission Flask app (:5000)
├── post_submission_demo/    # Post-submission FastAPI app (:8003)
├── mcp_server/              # MCP tool coordination server (:8001)
├── tools/                   # Shared utilities
│   ├── bedrock_llm.py       # Bedrock LLM wrapper (boto3 SigV4)
│   ├── bedrock_agent_integration.py  # Agent Core invocation
│   ├── s3_storage.py        # S3 read/write helpers
│   ├── x12_837p_builder.py  # ANSI X12 837P claim builder
│   └── era_processor.py     # ERA/835 parser
├── aws_setup/               # AWS provisioning scripts
├── config/settings.py       # Centralized config
├── data/                    # Local data + logs
└── docs/                    # Architecture diagrams
```

---

## Running Locally

### Prerequisites

- Python 3.11+
- AWS credentials with Bedrock and S3 access

### Setup

```bash
git clone <repo-url>
cd alpha

pip install -r requirements.txt
```

Create a `.env` file:

```env
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=alpha-claims-demo-390783052961
AWS_BEDROCK_MODEL_ID=us.amazon.nova-micro-v1:0

BEDROCK_AGENT_RISK_PREDICTOR=XLAYW801JO
BEDROCK_AGENT_AUTO_CORRECTOR=53KNQP1PMD
BEDROCK_AGENT_CLAIM_SUBMITTER=BSI3CF17OU
BEDROCK_AGENT_APPEAL_GENERATOR=S4YKZVC69F
BEDROCK_AGENT_RESUBMITTER=VCKGXKAZN0
BEDROCK_AGENT_FEEDBACK_LEARNER=WUKCB3RG8Z
```

### Start Everything

```bash
python start_all.py
```

Or individually:

```bash
# Pre-submission dashboard
python web_dashboard/api_server.py

# Post-submission appeals
python post_submission_demo/app.py

# MCP server
python mcp_server/main.py
```

### Access

| Service | URL |
|---------|-----|
| Pre-submission dashboard | http://localhost:5000 |
| Post-submission appeals | http://localhost:8003 |
| MCP server | http://localhost:8001 |
| Primary insurer API | http://localhost:8081 |
| Secondary insurer API | http://localhost:8082 |

---

## Screenshots

### Agent Activity Dashboard
![Agent Dashboard](Images/image-1.png)

### Claims Processing Interface
![Claims Interface](Images/image-2.png)

---

## Tech Stack

- **LangGraph** — multi-agent workflow orchestration
- **Amazon Bedrock** — AI inference (Nova Micro) + Agent Core
- **AWS Lambda** — serverless action groups per agent
- **Amazon S3** — persistent storage for all claim data
- **FastAPI + Flask** — backend APIs
- **MCP (Model Context Protocol)** — tool coordination layer
- **ANSI X12 837P / 835** — industry-standard claim and remittance formats

---

*Prototype built for AI for Bharat — Healthcare & Life Sciences domain.*
