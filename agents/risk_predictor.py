# agents/risk_predictor.py - Enhanced with MCP Integration

from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config.settings import Settings
from tools.logger import secure_log
from tools.csv_data_loader import patient_loader
from mcp_client import mcp_client
import json
import pandas as pd
import asyncio

# Setup LLM
llm = AzureChatOpenAI(
    azure_deployment=Settings.AZURE_OPENAI_DEPLOYMENT_NAME,
    api_key=Settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=Settings.AZURE_OPENAI_ENDPOINT,
    api_version=Settings.AZURE_OPENAI_API_VERSION,
    temperature=0.2,
    request_timeout=Settings.TIMEOUT
)

# Enhanced prompt with learning context
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert medical billing risk predictor with access to historical denial patterns. 
    
    Analyze the claim and predict denial risk considering:
    1. Patient demographics and medical history
    2. Insurance company specific patterns
    3. Previous denial reasons for similar claims
    4. CPT/ICD code combinations
    5. Claim amount and authorization requirements
    
    Historical denial patterns for this insurer:
    {denial_patterns}
    
    Return a JSON object with:
    - risk_score: float between 0-1 (0=low risk, 1=high risk)
    - issues: list of potential issues
    - recommendations: list of actions to reduce risk
    - confidence: float between 0-1 for prediction confidence
    """),
    ("human", """Analyze this claim for denial risk:
    
    Patient: {patient_name} (ID: {patient_id})
    Age: {age}, Gender: {gender}
    Diagnosis: {diagnosis} (ICD: {icd_code})
    Procedure: CPT {cpt_code}
    Amount: ${claim_amount}
    Insurance: {insurance_company}
    Prior Auth: {prior_auth}
    Medical History: {medical_history}
    Risk Factors: {risk_factors}
    Provider: {provider_name} (NPI: {provider_npi})
    
    Provide detailed risk analysis in JSON format.""")
])

async def run_risk_prediction(state: dict) -> dict:
    """Enhanced risk prediction with MCP-powered data sources"""
    
    claim_data = state.get("raw_data", {})
    
    try:
        # Get enhanced patient data via MCP
        patient_id = claim_data.get("patient_id", "")
        enhanced_patient = await mcp_client.get_patient_data(patient_id, include_medical_history=True)
        
        # Get insurance policy details via MCP
        insurance_company = claim_data.get("insurance_company", "")
        procedure_code = claim_data.get("cpt_code", "")
        diagnosis_code = claim_data.get("icd_code", "")
        claim_amount = claim_data.get("claim_amount", 0)
        
        policy_check = await mcp_client.check_insurance_policy(
            insurer=insurance_company,
            procedure_code=procedure_code,
            diagnosis_code=diagnosis_code,
            claim_amount=claim_amount
        )
        
        # Get denial patterns via MCP
        denial_analysis = await mcp_client.analyze_denial_patterns(
            insurer=insurance_company,
            procedure_code=procedure_code,
            time_period="90days"
        )
        
        # Get medical knowledge validation via MCP
        icd_validation = await mcp_client.query_medical_knowledge("icd_code", diagnosis_code)
        cpt_validation = await mcp_client.query_medical_knowledge("cpt_code", procedure_code)
        
        # Combine all MCP data sources
        mcp_data = {
            "enhanced_patient": enhanced_patient,
            "policy_check": policy_check,
            "denial_analysis": denial_analysis,
            "icd_validation": icd_validation,
            "cpt_validation": cpt_validation
        }
        
        # Enhanced denial patterns from MCP
        patterns_text = "\n".join([
            f"- {pattern.get('denial_reason', 'Unknown')} -> {pattern.get('learned_pattern', 'Pattern analysis')}"
            for pattern in (denial_analysis.get("patterns", []) if denial_analysis else [])[-5:]  # Last 5 patterns
        ])
        
        # Build comprehensive risk factors
        risk_factors = []
        
        # Add MCP-derived risk factors
        if policy_check and policy_check.get("prior_auth_required", False):
            risk_factors.append("Prior authorization required")
        
        if policy_check and policy_check.get("network_status") == "out-of-network":
            risk_factors.append("Out-of-network provider")
        
        if denial_analysis and denial_analysis.get("risk_score", 0) > 0.7:
            risk_factors.append("High historical denial rate for this procedure")
        
        if enhanced_patient and enhanced_patient.get("patient_data", {}).get("drug_interactions"):
            risk_factors.append("Potential drug interactions detected")
        
        # Enhanced claim data with MCP insights
        enhanced_claim = claim_data.copy()
        
        # Add patient data if available
        if enhanced_patient and enhanced_patient.get("patient_data"):
            patient_data = enhanced_patient["patient_data"]
            if patient_data:  # Additional safety check
                enhanced_claim.update({
                    "patient_name": patient_data.get("name", "Unknown"),
                    "age": patient_data.get("age", "Unknown"),
                    "gender": patient_data.get("gender", "Unknown"),
                })
        
        enhanced_claim.update({
            "mcp_risk_factors": risk_factors,
            "policy_coverage": policy_check.get("coverage_status", False) if policy_check else False,
            "historical_denial_rate": denial_analysis.get("denial_rate", 0) if denial_analysis else 0,
            "medical_necessity_score": icd_validation.get("confidence", 0.5) if icd_validation else 0.5,
            "procedure_validity_score": cpt_validation.get("confidence", 0.5) if cpt_validation else 0.5
        })
        
    except Exception as e:
        # Fallback to basic processing if MCP fails
        secure_log("risk_predictor", {"action": "mcp_error", "error": str(e)})
        
        enhanced_claim = claim_data.copy()
        enhanced_claim.update({
            "patient_name": "Unknown",
            "age": "Unknown", 
            "gender": "Unknown",
            "mcp_risk_factors": [],
            "policy_coverage": False,
            "historical_denial_rate": 0,
            "medical_necessity_score": 0.5,
            "procedure_validity_score": 0.5
        })
        
        patterns_text = "No historical patterns available"
        risk_factors = []
        
        # Format the prompt with MCP-enhanced data
        formatted_prompt = prompt.format_messages(
            patient_name=enhanced_claim.get("patient_name", "Unknown"),
            patient_id=enhanced_claim.get("patient_id", "Unknown"),
            age=enhanced_claim.get("age", "Unknown"),
            gender=enhanced_claim.get("gender", "Unknown"),
            diagnosis=enhanced_claim.get("diagnosis", "Unknown"),
            icd_code=enhanced_claim.get("icd_code", "Unknown"),
            cpt_code=enhanced_claim.get("cpt_code", "Unknown"),
            claim_amount=enhanced_claim.get("claim_amount", 0),
            insurance_company=insurance_company,
            prior_auth=enhanced_claim.get("prior_auth", "None"),
            medical_history=enhanced_claim.get("medical_history", "None"),
            risk_factors=", ".join(risk_factors) if risk_factors else "None detected",
            provider_name=enhanced_claim.get("provider_name", "Unknown"),
            provider_npi=enhanced_claim.get("provider_npi", "Unknown"),
            denial_patterns=patterns_text or "No historical patterns available"
        )
        
        # Get LLM response
        response = await llm.ainvoke(formatted_prompt)
        result = response.content
        
        # Parse JSON response
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            # Fallback parsing if JSON is malformed
            parsed = {
                "risk_score": 0.5,
                "issues": ["Unable to parse risk analysis"],
                "recommendations": ["Manual review required"],
                "confidence": 0.3
            }
        
        # Update state with enhanced risk information
        state["risk_score"] = float(parsed.get("risk_score", 0.5))
        state["issues"] = parsed.get("issues", [])
        state["recommendations"] = parsed.get("recommendations", [])
        state["confidence"] = float(parsed.get("confidence", 0.5))
        
        # Add MCP-enhanced data to state
        state["mcp_data"] = mcp_data
        state["policy_coverage"] = policy_check.get("coverage_status", False)
        state["historical_denial_rate"] = denial_analysis.get("denial_rate", 0)
        state["enhanced_patient_data"] = enhanced_patient
        
        # Set processing status
        state["final_status"] = "risk_assessed"
        
        # Enhanced logging
        state["log"].append(
            f"[RiskPredictor-MCP] Risk: {state['risk_score']:.2f}, "
            f"Confidence: {state['confidence']:.2f}, "
            f"Issues: {len(state['issues'])}, "
            f"Recommendations: {len(state['recommendations'])}, "
            f"Policy Coverage: {state['policy_coverage']}, "
            f"Historical Denial Rate: {state['historical_denial_rate']:.2f}"
        )
        
        secure_log("RiskPredictor-MCP", {
            "claim_id": state.get("claim_id"),
            "risk_score": state["risk_score"],
            "issues": state["issues"],
            "recommendations": state["recommendations"],
            "confidence": state["confidence"],
            "policy_coverage": state["policy_coverage"],
            "historical_denial_rate": state["historical_denial_rate"],
            "mcp_data_sources": list(mcp_data.keys()),
            "final_status": state.get("final_status", "processing"),
            "log": state.get("log", [])
        })
        
        return state

    except Exception as e:
        state["log"].append(f"[RiskPredictor-MCP] Error: {str(e)}")
        # Fallback risk assessment
        state["risk_score"] = 0.5
        state["issues"] = [f"Risk prediction failed: {str(e)}"]
        state["recommendations"] = ["Manual review required"]
        state["confidence"] = 0.1
        return state
