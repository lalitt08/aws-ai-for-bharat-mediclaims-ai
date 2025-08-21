# agents/appeal_generator.py

from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config.settings import Settings
from tools.formatter import generate_appeal_pdf
from tools.logger import secure_log

# Setup Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_endpoint=Settings.AZURE_OPENAI_ENDPOINT,
    azure_deployment=Settings.AZURE_OPENAI_DEPLOYMENT_NAME,
    openai_api_key=Settings.AZURE_OPENAI_API_KEY,
    openai_api_version=Settings.AZURE_OPENAI_API_VERSION,
    temperature=0.3,
    request_timeout=Settings.TIMEOUT
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert medical appeal writer. Use the provided claim to draft a formal appeal."),
    ("human", "Claim Info:\n{claim_data}\n\nRejection Reason:\n{rejection_reason}\n\nWrite an appeal letter justifying the treatment.")
])

async def run_appeal_generation(state: dict) -> dict:
    """Enhanced Appeal Generator with intelligent data requirement analysis"""
    claim_data = state.get("corrected_data") or state.get("raw_data")
    result = state.get("submission_result", {})
    reason = result.get("reason", "No reason provided")
    denial_info = result.get("denial_info", {})

    try:
        # Add detailed logging for UI activity tracking
        state["log"].append("[AppealGenerator] Analyzing denial reason and determining if patient data updates are needed")
        state["log"].append("[AppealGenerator] Azure OpenAI generating intelligent appeal strategy")
        
        # Analyze if this requires patient data updates in OpenEMR
        requires_patient_update = analyze_data_requirements(denial_info, claim_data)
        
        if requires_patient_update['needed']:
            state["log"].append(f"[AppealGenerator] 🔍 Patient data update required: {', '.join(requires_patient_update['missing_fields'])}")
            state["log"].append("[AppealGenerator] Will notify user to update OpenEMR patient records")
            
            # Store the requirement info for the UI
            state["patient_update_required"] = requires_patient_update
            state["appeal_packet"] = "Patient data update required before appeal submission"
            state["log"].append("[AppealGenerator] Appeal preparation paused - waiting for OpenEMR data update")
            state["final_status"] = "awaiting_patient_data_update"
            
        else:
            state["log"].append("[AppealGenerator] Sufficient data available, proceeding with AI appeal generation")
            
            formatted_prompt = prompt.format_messages(
                claim_data=str(claim_data),
                rejection_reason=reason
            )
            response = await llm.ainvoke(formatted_prompt)
            appeal_text = response.content.strip()

            # Optional: Convert to PDF (handle errors gracefully)
            try:
                appeal_pdf_path = generate_appeal_pdf(
                    claim_id=state["claim_id"],
                    appeal_text=appeal_text
                )
                state["appeal_packet"] = appeal_pdf_path
                state["log"].append(f"[AppealGenerator] Appeal created with PDF: {appeal_pdf_path}")
            except Exception as pdf_error:
                # Don't let PDF generation failure block the workflow
                state["appeal_packet"] = f"Appeal text generated (PDF creation failed: {str(pdf_error)})"
                state["log"].append(f"[AppealGenerator] Appeal created (PDF generation failed)")
            
            # Store the appeal text even if PDF fails
            state["appeal_text"] = appeal_text
            state["log"].append(f"[AppealGenerator] ✅ AI-powered appeal completed - ready for automatic resubmission")
            
            # Set processing status
            state["final_status"] = "appeal_generated"
        
        secure_log("AppealGenerator", state)
        return state

    except Exception as e:
        state["log"].append(f"[AppealGenerator] Error: {str(e)}")
        raise


def analyze_data_requirements(denial_info: dict, claim_data: dict) -> dict:
    """Analyze if patient data updates are needed in OpenEMR"""
    
    missing_fields = []
    requirements = denial_info.get('required_items', [])
    reason = denial_info.get('reason', '').lower()
    details = denial_info.get('details', '').lower()
    
    # Categories that CAN be auto-appealed (no patient data update needed)
    auto_appealable_categories = [
        'authorization', 'coding', 'modifier', 'credentials', 'enrollment',
        'icd-10', 'cpt', 'npi', 'provider', 'billing', 'service level'
    ]
    
    # Check if this is an auto-appealable denial
    is_auto_appealable = any(term in reason or term in details for term in auto_appealable_categories)
    
    # Specific checks for issues that require patient data updates
    needs_patient_data = False
    
    for requirement in requirements:
        req_lower = requirement.lower()
        
        # Only flag as needing patient data if it's CORE patient medical information
        # that should come from the patient's medical record
        if any(term in req_lower for term in [
            'medical history', 'patient history', 'previous medical', 'past medical',
            'symptoms', 'patient symptoms', 'chief complaint',
            'allergies', 'medication history', 'family history'
        ]):
            if not claim_data.get('medical_history') or claim_data.get('medical_history') in ['None', 'No history available']:
                missing_fields.append('Complete medical history from patient records')
                needs_patient_data = True
        
        # Patient demographics that should be in OpenEMR
        elif any(term in req_lower for term in ['patient demographics', 'patient information', 'patient details']):
            demo_missing = []
            if not claim_data.get('date_of_birth'):
                demo_missing.append('date of birth')
            if not claim_data.get('gender'):
                demo_missing.append('gender')
            if demo_missing:
                missing_fields.append(f"Patient demographics: {', '.join(demo_missing)}")
                needs_patient_data = True
    
    # CRITICAL FIX: If it's auto-appealable and no core patient data is missing,
    # proceed with automated appeal
    if is_auto_appealable and not needs_patient_data:
        return {
            'needed': False,  # No patient data update needed
            'missing_fields': [],
            'can_proceed_without': True,
            'recommendation': 'Auto-appealable - proceeding with AI appeal generation',
            'appeal_type': 'automated'
        }
    
    # Only require patient data updates for genuine patient medical information gaps
    return {
        'needed': needs_patient_data,
        'missing_fields': missing_fields,
        'can_proceed_without': not needs_patient_data,
        'recommendation': 'Update patient records in OpenEMR with missing information' if needs_patient_data else 'Sufficient data available for appeal',
        'appeal_type': 'requires_patient_data' if needs_patient_data else 'automated'
    }
