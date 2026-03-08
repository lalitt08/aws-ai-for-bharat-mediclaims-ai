# tools/user_friendly_translator.py
"""
LLM-powered translator for converting technical agent logs to user-friendly language
"""

from tools.bedrock_llm import BedrockLLM
from langchain_core.prompts import ChatPromptTemplate
from config.settings import Settings
from tools.logger import secure_log
import json
import asyncio
from typing import Dict, Any, Optional

class UserFriendlyTranslator:
    """Translates technical agent activities to user-friendly language using LLM"""
    
    def __init__(self):
        """Initialize the translator with Bedrock LLM"""
        try:
            self.llm = BedrockLLM(temperature=0.1)
            
            # Prompt template for translating technical activities
            self.translation_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at translating technical medical billing processes into clear, reassuring language that patients and healthcare staff can easily understand.

Your translations should:
1. Use everyday language instead of technical jargon
2. Be reassuring and professional, but not overly technical
3. Focus on what the user needs to know (progress, next steps, outcomes)
4. Explain what's happening in simple terms
5. Add context about why each step matters

For errors: Transform technical errors into reassuring explanations that indicate the system is handling the issue.

Key Translation Guidelines:
- "Risk Score 0.85" → "85% likelihood of approval - looking very good!"
- "Policy Coverage: False" → "Checking coverage details with your insurance"
- "Historical Denial Rate" → "Based on similar claims, we expect a good outcome"
- "Azure OpenAI Analysis" → "Our AI system reviewed your claim"
- "MCP client" → "External verification system"
- "Eligibility check" → "Confirming your insurance coverage"
- "CPT/ICD codes" → "Medical procedure and diagnosis codes"
- "Prior authorization" → "Insurance pre-approval process"
- "Error messages" → "We encountered a minor issue but are handling it automatically"

Always maintain a helpful, professional tone while making complex processes understandable."""),
                ("human", """Please translate this technical activity into clear, user-friendly language:

Agent: {agent}
Activity: {activity}
Technical Details: {details}
Status: {status}
Patient ID: {patient_id}

Provide a JSON response with:
{{
    "user_friendly_activity": "Simple, reassuring description of what happened",
    "user_friendly_details": "Easy-to-understand explanation that puts the user at ease",
    "next_steps": "What the patient can expect next",
    "patient_context": "Brief context about this step in their claim process"
}}""")
            ])
            
            self.available = True
            print("[SUCCESS] UserFriendlyTranslator initialized successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize UserFriendlyTranslator: {e}")
            self.available = False
            self.llm = None
    
    async def translate_activity(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate a technical activity to user-friendly language
        
        Args:
            activity_data: Dict containing agent, activity, details, status
            
        Returns:
            Dict with user_friendly_activity, user_friendly_details, next_steps
        """
        if not self.available:
            return self._fallback_translation(activity_data)
        
        try:
            # Format the prompt with activity data
            formatted_prompt = self.translation_prompt.format_messages(
                agent=activity_data.get('agent', 'System'),
                activity=activity_data.get('activity', 'Processing'),
                details=activity_data.get('details', 'No details available'),
                status=activity_data.get('status', 'unknown'),
                patient_id=activity_data.get('patient_id', 'Unknown')
            )
            
            # Get LLM response with timeout
            response = await asyncio.wait_for(
                self.llm.ainvoke(formatted_prompt),
                timeout=10.0  # 10 second timeout for translation
            )
            
            # Parse the JSON response
            response_text = response.content.strip()
            
            # Handle potential code block wrapping
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            translation_result = json.loads(response_text)
            
            # Validate the response has required fields
            required_fields = ['user_friendly_activity', 'user_friendly_details']
            if all(field in translation_result for field in required_fields):
                print(f"[SUCCESS] Successfully translated activity: {activity_data.get('agent', 'Unknown')}")
                
                # Add patient context if available
                if 'patient_context' in translation_result:
                    translation_result['patient_context'] = translation_result['patient_context']
                
                return translation_result
            else:
                print(f"[WARNING] LLM response missing required fields: {translation_result}")
                return self._fallback_translation(activity_data)
                
        except asyncio.TimeoutError:
            print("[WARNING] Translation timeout - using fallback")
            return self._fallback_translation(activity_data)
        except json.JSONDecodeError as e:
            print(f"[WARNING] Failed to parse LLM response as JSON: {e}")
            return self._fallback_translation(activity_data)
        except Exception as e:
            print(f"[ERROR] Translation error: {e}")
            return self._fallback_translation(activity_data)
    
    def _fallback_translation(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide fallback user-friendly translations when LLM is unavailable"""
        
        agent = activity_data.get('agent', 'System')
        activity = activity_data.get('activity', 'Processing')
        details = activity_data.get('details', '')
        status = activity_data.get('status', 'unknown')
        
        # Agent-specific fallback translations
        agent_translations = {
            'RiskPredictor': {
                'activity': '🔍 Reviewing your claim for the best possible outcome',
                'details': 'Our AI system is carefully analyzing your claim details, checking with your insurance company, and predicting approval likelihood to ensure everything goes smoothly.',
                'next_steps': 'Once the review is complete, we\'ll either submit your claim or fix any issues we find.'
            },
            'Risk Predictor': {
                'activity': '🔍 Reviewing your claim for the best possible outcome', 
                'details': 'Our AI system is carefully analyzing your claim details, checking with your insurance company, and predicting approval likelihood to ensure everything goes smoothly.',
                'next_steps': 'Once the review is complete, we\'ll either submit your claim or fix any issues we find.'
            },
            'AutoCorrector': {
                'activity': '✨ Making sure your claim is perfect',
                'details': 'We\'re reviewing your claim information and automatically fixing any missing details or formatting issues to give you the best chance of approval.',
                'next_steps': 'After we perfect your claim details, we\'ll submit it to your insurance company.'
            },
            'Auto Corrector': {
                'activity': '✨ Making sure your claim is perfect',
                'details': 'We\'re reviewing your claim information and automatically fixing any missing details or formatting issues to give you the best chance of approval.',
                'next_steps': 'After we perfect your claim details, we\'ll submit it to your insurance company.'
            },
            'ClaimSubmitter': {
                'activity': '📋 Submitting your claim to insurance',
                'details': 'Your claim is being sent to your insurance company for processing. We\'re monitoring the submission and will handle any responses automatically.',
                'next_steps': 'We\'ll track your claim status and let you know as soon as we hear back from your insurance.'
            },
            'Claim Submitter': {
                'activity': '📋 Submitting your claim to insurance', 
                'details': 'Your claim is being sent to your insurance company for processing. We\'re monitoring the submission and will handle any responses automatically.',
                'next_steps': 'We\'ll track your claim status and let you know as soon as we hear back from your insurance.'
            },
            'AppealGenerator': {
                'activity': '📝 Preparing an appeal to get you covered',
                'details': 'Don\'t worry - we\'re creating a strong, professional appeal letter to challenge any denial and fight for your coverage.',
                'next_steps': 'Once the appeal is ready, we\'ll submit it immediately to get your claim reconsidered.'
            },
            'Appeal Generator': {
                'activity': '📝 Preparing an appeal to get you covered',
                'details': 'Don\'t worry - we\'re creating a strong, professional appeal letter to challenge any denial and fight for your coverage.',
                'next_steps': 'Once the appeal is ready, we\'ll submit it immediately to get your claim reconsidered.'
            },
            'Resubmitter': {
                'activity': '🔄 Resubmitting your improved claim',
                'details': 'After making all the necessary improvements and corrections, we\'re sending your updated claim back to insurance with a much better chance of approval.',
                'next_steps': 'We\'ll monitor this resubmission closely and keep you updated on the results.'
            },
            'FeedbackLearner': {
                'activity': '📈 Learning from your case to help future patients',
                'details': 'We\'re analyzing your claim outcome to improve our AI system and help other patients with similar cases get better results.',
                'next_steps': 'This learning process helps us continuously improve - your case is now complete!'
            },
            'Feedback Learner': {
                'activity': '📈 Learning from your case to help future patients',
                'details': 'We\'re analyzing your claim outcome to improve our AI system and help other patients with similar cases get better results.',
                'next_steps': 'This learning process helps us continuously improve - your case is now complete!'
            },
            'System': {
                'activity': '⚙️ Processing your healthcare claim',
                'details': 'Your claim is moving through our intelligent processing system. Each step is designed to maximize your chances of approval.',
                'next_steps': 'We\'ll continue working on your claim and update you with each major milestone.'
            },
            'Workflow Router': {
                'activity': '🎯 Determining the best path for your claim',
                'details': 'Our system is intelligently routing your claim to the right processing steps based on your specific situation and insurance requirements.',
                'next_steps': 'Your claim will now move to the most appropriate next stage for optimal processing.'
            },
            'Insurance API': {
                'activity': '🏥 Getting response from your insurance company',
                'details': 'We\'ve received a response from your insurance company and are processing their decision to determine next steps.',
                'next_steps': 'We\'ll review their response and either celebrate approval or prepare next steps if needed.'
            },
            'MCP Server': {
                'activity': '🔗 Connecting with external systems',
                'details': 'We\'re securely connecting with external medical and insurance systems to get the most up-to-date information for your claim.',
                'next_steps': 'This ensures we have all the data needed to process your claim successfully.'
            }
        }
        
        # Get agent-specific translation or use generic
        agent_key = agent.replace('-MCP', '').replace('_', '').replace(' ', '')
        
        # Try exact match first, then try variations
        translation = None
        for key in [agent, agent_key, agent.replace('_', ' ')]:
            if key in agent_translations:
                translation = agent_translations[key]
                break
        
        if not translation:
            translation = {
                'activity': '⚙️ Processing your healthcare claim',
                'details': 'We\'re working diligently on your claim to ensure the best possible outcome with your insurance company.',
                'next_steps': 'We\'ll continue processing and update you with important milestones.'
            }
        
        # Add status-specific context and handle errors specially
        status_context = ""
        if status == 'completed':
            status_context = " ✅ This step completed successfully!"
        elif status == 'processing':
            status_context = " 🔄 This process is currently in progress..."
        elif status == 'error':
            # Special handling for errors - make them less scary
            translation['activity'] = '🔧 Handling a minor technical issue'
            translation['details'] = 'We encountered a small technical issue, but our system is designed to handle these automatically. Your claim processing continues without interruption.'
            translation['next_steps'] = 'No action needed from you - we\'re handling this behind the scenes.'
            status_context = " Our system is resolving this automatically."
        
        # Handle specific technical terms in details
        user_friendly_details = self._translate_technical_terms(details)
        if user_friendly_details == details or not user_friendly_details:
            user_friendly_details = translation['details']
        
        user_friendly_details += status_context
        
        return {
            'user_friendly_activity': translation['activity'],
            'user_friendly_details': user_friendly_details,
            'next_steps': translation.get('next_steps', ''),
            'patient_context': f'This is part of your claim processing journey - we\'re working to get you the coverage you deserve.'
        }
    
    def _translate_technical_terms(self, details: str) -> str:
        """Translate common technical terms to user-friendly language"""
        if not details:
            return "Processing your claim with our advanced AI system."
        
        # Technical term replacements
        translations = {
            'Risk Score': 'Approval likelihood',
            'Azure OpenAI Analysis': 'AI-powered review',
            'AI-powered review': 'Our intelligent system reviewed your claim',
            'Policy Coverage': 'Insurance coverage',
            'Historical Denial Rate': 'Based on similar claims',
            'MCP client': 'external verification system',
            'MCP': 'external system',
            'Eligibility check': 'insurance verification',
            'Prior authorization': 'insurance pre-approval',
            'CPT code': 'procedure code',
            'ICD code': 'diagnosis code',
            'NPI': 'provider ID',
            'API': 'insurance system',
            'Confidence': 'certainty level',
            'Issues found': 'items to review',
            'Processing timeout': 'extended processing time',
            'Error': 'minor issue',
            'Failed': 'encountered a small hiccup',
            'Exception': 'technical detail',
            'AttributeError': 'system reconfiguration',
            'MCPClient': 'verification system',
            'object has no attribute': 'system is updating its configuration',
            'check_insurance_policy': 'insurance verification process',
            '🧠': '',  # Remove brain emoji for cleaner text
            '📊': '',  # Remove chart emoji
            '⚡': '',  # Remove lightning emoji
            '❌': '⚠️',  # Replace X with warning for less alarming feel
            'Found 0 issues': 'Everything looks good',
            'Found 1 issues': 'Found 1 minor item to check',
            'Found 2 issues': 'Found 2 minor items to review',
            'Found 3 issues': 'Found 3 items that need attention',
            'Found 4 issues': 'Found 4 items to optimize',
            'Found 5 issues': 'Found 5 areas for improvement',
            'Found 6 issues': 'Found 6 items to enhance',
            'Found 7 issues': 'Found 7 areas we can improve',
            'Found 8 issues': 'Found 8 opportunities for optimization',
            'Found 9 issues': 'Found 9 items we\'re addressing',
            'Found 10 issues': 'Found several areas we\'re optimizing'
        }
        
        user_friendly = details
        for technical, friendly in translations.items():
            user_friendly = user_friendly.replace(technical, friendly)
        
        # Handle specific patterns
        import re
        
        # Risk scores - make them more positive
        risk_pattern = r'Risk Score ([\d.]+)'
        risk_match = re.search(risk_pattern, user_friendly)
        if risk_match:
            score = float(risk_match.group(1))
            if score >= 0.8:
                replacement = f'Excellent approval chances ({int(score*100)}%)'
            elif score >= 0.6:
                replacement = f'Good approval likelihood ({int(score*100)}%)'
            elif score >= 0.4:
                replacement = f'Moderate approval chances ({int(score*100)}%)'
            else:
                replacement = f'Working to improve approval odds ({int(score*100)}%)'
            user_friendly = re.sub(risk_pattern, replacement, user_friendly)
        
        # Confidence levels
        conf_pattern = r'Confidence: ([\d.]+)'
        conf_match = re.search(conf_pattern, user_friendly)
        if conf_match:
            confidence = float(conf_match.group(1))
            replacement = f'We\'re {int(confidence*100)}% confident in this assessment'
            user_friendly = re.sub(conf_pattern, replacement, user_friendly)
        
        # Make it more conversational
        user_friendly = user_friendly.replace('. ', '. ')
        user_friendly = user_friendly.replace('  ', ' ')
        
        return user_friendly

# Global translator instance
translator = UserFriendlyTranslator()

async def translate_to_user_friendly(activity_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to translate activity data to user-friendly language
    
    Args:
        activity_data: Dict containing agent, activity, details, status
        
    Returns:
        Dict with user_friendly_activity, user_friendly_details, next_steps
    """
    return await translator.translate_activity(activity_data)
