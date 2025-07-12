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
    claim_data = state.get("corrected_data") or state.get("raw_data")
    result = state.get("submission_result", {})
    reason = result.get("reason", "No reason provided")

    try:
        formatted_prompt = prompt.format_messages(
            claim_data=str(claim_data),
            rejection_reason=reason
        )
        response = await llm.ainvoke(formatted_prompt)
        appeal_text = response.content.strip()

        # Optional: Convert to PDF
        appeal_pdf_path = generate_appeal_pdf(
            claim_id=state["claim_id"],
            appeal_text=appeal_text
        )

        state["appeal_packet"] = appeal_pdf_path
        state["log"].append(f"[AppealGenerator] Appeal created at {appeal_pdf_path}")
        
        # Set processing status
        state["final_status"] = "appeal_generated"
        
        secure_log("AppealGenerator", state)
        return state

    except Exception as e:
        state["log"].append(f"[AppealGenerator] Error: {str(e)}")
        raise
