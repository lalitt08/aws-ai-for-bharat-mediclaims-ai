# MediClaims AI

A prototype built for **AI for Bharat Hackathon** under the **Healthcare & Life Sciences** domain.

---

## What It Does

Healthcare providers lose revenue every year due to insurance claim denials caused by missing data, wrong codes, or incomplete documentation. MediClaims AI automates the entire claims lifecycle using six AI agents that work together — from fetching patient data, predicting denials, correcting errors, submitting claims, generating appeals, and learning from outcomes.

---

## The Six AI Agents

| Agent | What It Does |
|-------|-------------|
| Risk Predictor | Scores each claim for denial probability before submission |
| Auto Corrector | Fixes missing fields, validates medical codes, adds prior auth |
| Claim Submitter | Submits claims to insurers in ANSI X12 837P format |
| Appeal Generator | Writes an AI-generated appeal letter based on the denial reason |
| Resubmitter | Resubmits the corrected claim with the appeal attached |
| Feedback Learner | Learns from every outcome to improve future predictions |

---

## AWS Services Used

| Service | Purpose |
|---------|---------|
| **EC2** (t3.medium) | Hosts Flask :5000 (pre-submission), FastAPI :8003 (post-submission), MCP :8001, nginx |
| **Application Load Balancer** | Single public URL, routes traffic to Flask and FastAPI |
| **Bedrock Agent Core** | Six individual AI agents, each with its own ID and action group |
| **Bedrock — Nova Micro** | Foundation model used by all agents (`us.amazon.nova-micro-v1:0`) |
| **Lambda** | Six functions as action groups — implement real tool logic per agent |
| **S3** | Stores patient data, claims, ERA files, appeal PDFs, X12 transactions, logs |
| **IAM** | `MediClaimsEC2Role` and `BedrockAgentsClaimsRole` for secure access |

---

## Live URLs

| | URL |
|-|-----|
| Pre-submission | http://mediclaims-alb-640335154.us-east-1.elb.amazonaws.com/ |
| Post-submission | http://mediclaims-alb-640335154.us-east-1.elb.amazonaws.com/appeals/ |

---

## Run Locally

```bash
pip install -r requirements.txt
# add AWS credentials to .env
python start_all.py
```

| Service | URL |
|---------|-----|
| Pre-submission dashboard | http://localhost:5000 |
| Post-submission appeals | http://localhost:8003 |

---

## Screenshots

![Agent Dashboard](Images/image-1.png)
![Claims Interface](Images/image-2.png)

---

*Prototype — AI for Bharat Hackathon 2026*
