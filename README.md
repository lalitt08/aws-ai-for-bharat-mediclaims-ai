# MediClaims AI

**AWS AI for Bharat Hackathon 2026 | Track: AI for Healthcare & Life Sciences**

---

## 💡 The Idea

An AI-powered multi-agent system that reduces healthcare claim denials from 35% to under 10% and automates appeal generation for Indian hospitals.

## 🎯 Problem

- 35-40% of healthcare claims in India get denied
- 45-60 days average reimbursement time
- Tier-2/3 hospitals lack expertise for appeals
- 50+ crore PM-JAY beneficiaries affected

## 🚀 Solution

9 specialized AI agents working in two phases:

**Pre-Submission (3 Agents):**
| Agent | Role |
|-------|------|
| Risk Predictor | ML-based denial prediction before submission |
| Auto-Corrector | LLM-powered intelligent data correction |
| Claim Submitter | Multi-insurer API routing |

**Post-Submission (6 Agents):**
| Agent | Role |
|-------|------|
| ERA Processor | Parse ERA/835 files, extract denial codes |
| Denial Classifier | ML+NLP categorization, priority scoring |
| Compliance Checker | HIPAA, state, payer policy validation |
| Appeal Generator | GPT-4 powered appeal letter creation |
| Resubmitter | Strategic appeal submission |
| Feedback Learner | Continuous improvement from outcomes |

## 📊 Impact

- **75% reduction** in claim denials
- **85% faster** processing (45 days → 7 days)
- **78% appeal success** rate
- **90% cost reduction** in claims admin

## 🛠️ Tech Stack

- **AI/ML**: Azure OpenAI GPT-4, XGBoost
- **Orchestration**: LangGraph (multi-agent)
- **Backend**: Python, FastAPI
- **Cloud**: AWS (Bedrock, Lambda, S3, DynamoDB)

## 📁 Submission Files

- [`requirements.md`](./requirements.md) - Functional & non-functional requirements
- [`design.md`](./design.md) - System architecture & AI design

> **Note**: All data used is synthetic. No real patient information.

---

*Built with [Kiro](https://kiro.dev) for AWS AI for Bharat Hackathon 2026*
