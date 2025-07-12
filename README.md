# 🏥 Healthcare MediClaims AI - Agentic System

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.40+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![OpenAI](https://img.shields.io/badge/OpenAI-1.0+-red.svg)](https://openai.com/)

## 🎯 Project Overview

This is an **AI-powered system** designed to help hospitals and healthcare providers handle medical claim denials more efficiently. The system uses multiple AI agents working together to:

- **Predict** the likelihood of claim denials before submission
- **Automatically correct** common issues in claims
- **Submit** corrected claims to insurance companies
- **Generate appeals** for denied claims and resubmit them
- **Learn** from feedback to improve future performance

**The Goal**: Reduce manual effort, improve approval rates, and speed up the overall claim resolution process while continuously learning and improving over time.

## � Project Presentation

🎯 **[View Detailed Project Presentation](https://gamma.app/docs/Agentic-AI-for-Claims-Clinical-Trial-Billing-vu8omp3vx7awy85?mode=doc)**

*Access the comprehensive presentation covering the Agentic AI system for Claims & Clinical Trial Billing, including system architecture, AI workflows, and business benefits.*

## �📸 System Screenshots

### Dashboard Overview
*[Screenshot placeholder - Main dashboard showing claim status and metrics]*

### Claim Processing Flow
*[Screenshot placeholder - Visual representation of the AI agent workflow]*

### Risk Prediction Interface
*[Screenshot placeholder - Risk assessment and prediction results]*

### Appeal Generation
*[Screenshot placeholder - Auto-generated appeal documents]*

## 🏗️ System Architecture

The system consists of multiple AI agents working together:

### 🤖 AI Agents
- **Risk Predictor**: Analyzes claims and predicts denial probability
- **Auto Corrector**: Fixes common issues in claims automatically
- **Claim Submitter**: Handles submission to insurance APIs
- **Appeal Generator**: Creates compelling appeal documents
- **Resubmitter**: Manages resubmission of corrected claims
- **Feedback Learner**: Learns from outcomes to improve future performance

### 🔧 Core Components
- **MCP Server**: AI tools and model context protocol integration
- **Orchestrator**: Coordinates multi-agent workflows
- **Web Dashboard**: User interface for monitoring and control
- **Insurance APIs**: Integration with major insurance providers
- **Data Storage**: Patient data, claim history, and learning logs

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd agentic-claims-ai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   AZURE_OPENAI_API_KEY=your_azure_openai_key_here (optional)
   MCP_SERVER_PORT=3000
   API_SERVER_PORT=5000
   ```

### Running the System

#### Option 1: Using the Batch File (Windows)
```bash
start.bat
```

#### Option 2: Using Python Script
```bash
python start_agentic_system.py
```

#### Option 3: Manual Component Startup
```bash
# Start MCP Server
python mcp_server/main.py

# Start API Server (in another terminal)
python web_dashboard/api_server.py

# Start Orchestrator (in another terminal)
python orchestrator/orchestrator.py
```

### Accessing the System

Once started, you can access:
- **Web Dashboard**: http://localhost:5000
- **API Documentation**: http://localhost:5000/docs
- **MCP Server**: http://localhost:3000

## 📊 Features

### 🔍 Claim Risk Assessment
- AI-powered risk prediction before submission
- Historical analysis of denial patterns
- Confidence scoring for each prediction

### 🔧 Automatic Correction
- Common error detection and fixing
- Medical coding validation
- Documentation completeness checks

### 📤 Intelligent Submission
- Multi-insurer API integration
- Optimal timing for submissions
- Retry logic with exponential backoff

### 📋 Appeal Generation
- AI-generated appeal letters
- Medical necessity documentation
- Regulatory compliance checks

### 📈 Performance Tracking
- Success rate monitoring
- Learning from feedback
- Continuous improvement metrics

## 🛠️ Configuration

### Agent Settings
Configure individual agents in `config/settings.py`:
```python
# Risk prediction thresholds
RISK_THRESHOLD = 0.7
CONFIDENCE_THRESHOLD = 0.8

# Auto-correction settings
ENABLE_AUTO_CORRECTION = True
MAX_CORRECTION_ATTEMPTS = 3

# Learning parameters
LEARNING_RATE = 0.01
FEEDBACK_WEIGHT = 0.5
```

### Insurance API Configuration
Update insurance provider settings in the respective API files:
- `tools/insurer_api_primary.py` - BlueCross, Aetna
- `tools/insurer_api_secondary.py` - Cigna, United Healthcare

## 📁 Project Structure

```
agentic-claims-ai/
├── agents/                 # AI agent implementations
│   ├── risk_predictor.py  # Claim risk assessment
│   ├── auto_corrector.py  # Automatic error correction
│   ├── claim_submitter.py # Claim submission logic
│   ├── appeal_generator.py # Appeal document creation
│   ├── resubmitter.py     # Resubmission handling
│   └── feedback_learner.py # Learning from outcomes
├── config/                 # Configuration files
│   └── settings.py        # System settings
├── data/                  # Data storage
│   ├── patients.csv       # Patient information
│   ├── claim_status.json  # Claim tracking
│   └── training/          # Learning data
├── graph/                 # LangGraph workflow definitions
│   ├── claim_flow.py      # Main claim processing flow
│   └── nodes.py           # Individual workflow nodes
├── mcp_server/            # Model Context Protocol server
│   └── main.py           # MCP server implementation
├── orchestrator/          # Agent coordination
│   ├── orchestrator.py   # Main orchestration logic
│   └── mcp_client.py     # MCP client integration
├── tools/                 # Utility tools and APIs
│   ├── insurer_api.py    # Insurance API interfaces
│   ├── medical_knowledge_base.py # Medical coding
│   └── validators.py     # Data validation
├── web_dashboard/         # Web interface
│   ├── dashboard.html    # Main dashboard
│   ├── api_server.py     # FastAPI server
│   └── styles.css        # Dashboard styling
├── requirements.txt       # Python dependencies
└── start_agentic_system.py # System startup script
```

## 🔌 API Integration

### Supported Insurance Providers
- **BlueCross BlueShield**
- **Aetna**
- **Cigna**
- **United Healthcare**
- **Custom API endpoints**

### Adding New Insurance Providers
1. Create a new API class in `tools/`
2. Implement the required methods:
   - `submit_claim()`
   - `check_status()`
   - `submit_appeal()`
3. Register the provider in the orchestrator

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run specific test categories:
```bash
# Test AI agents
pytest tests/test_agents.py

# Test API integrations
pytest tests/test_apis.py

# Test workflows
pytest tests/test_workflows.py
```

## 📝 Logging

The system provides comprehensive logging:
- **Application logs**: `data/logs/application.log`
- **Agent activity**: `data/logs/agents.log`
- **API interactions**: `data/logs/api.log`
- **Learning data**: `data/training/feedback_log.jsonl`

## 🔒 Security & Compliance

- **HIPAA Compliance**: Patient data encryption and access controls
- **API Security**: Token-based authentication for insurance APIs
- **Data Privacy**: Local data storage with secure handling
- **Audit Trails**: Complete logging of all system activities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- **Documentation**: Check the `/docs` folder for detailed guides
- **Issues**: Open an issue on GitHub
- **Email**: [Your contact email]

## 🎯 Roadmap

### Phase 1 (Current)
- ✅ Core AI agents implementation
- ✅ MCP integration
- ✅ Basic web dashboard
- ✅ Insurance API integration

### Phase 2 (Next)
- 🔄 Advanced machine learning models
- 🔄 Real-time claim monitoring
- 🔄 Mobile app interface
- 🔄 Advanced analytics dashboard

### Phase 3 (Future)
- 📋 Integration with hospital EMR systems
- 📋 Blockchain-based claim verification
- 📋 Multi-language support
- 📋 Enterprise deployment options

---

**Made with ❤️ for healthcare providers** - Transforming claim processing through AI automation
