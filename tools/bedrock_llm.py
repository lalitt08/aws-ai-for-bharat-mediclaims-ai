"""
BedrockLLM — drop-in replacement for AzureChatOpenAI
Uses AWS Bedrock via boto3 (IAM key-based SigV4 auth).
Supports both sync invoke() and async ainvoke() with same interface.
"""

import os
import json
import asyncio
import logging
import boto3
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BEDROCK_REGION   = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("AWS_BEDROCK_MODEL_ID", "us.amazon.nova-micro-v1:0")


class _BedrockResponse:
    """Mimics langchain AIMessage so .content works everywhere."""
    def __init__(self, content: str):
        self.content = content


class BedrockLLM:
    """
    Drop-in replacement for AzureChatOpenAI / ChatOpenAI.
    Uses boto3 bedrock-runtime with IAM key-based auth (AWS_ACCESS_KEY_ID /
    AWS_SECRET_ACCESS_KEY from environment / ~/.aws/credentials).

    Usage (identical to AzureChatOpenAI):
        llm = BedrockLLM(temperature=0.3)
        response = await llm.ainvoke(messages)
        print(response.content)
    """

    def __init__(
        self,
        model_id: str = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        region: str = None,
        # Accept and ignore legacy / Azure-specific kwargs for seamless swap
        api_key: str = None,
        **kwargs,
    ):
        self.model_id    = model_id or BEDROCK_MODEL_ID
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.region      = region or BEDROCK_REGION
        self._family     = self._detect_family()
        self._client     = boto3.client("bedrock-runtime", region_name=self.region)

    def _detect_family(self) -> str:
        mid = self.model_id.lower()
        if "llama"  in mid: return "llama"
        if "nova"   in mid: return "nova"
        if "titan"  in mid: return "titan"
        if "claude" in mid: return "claude"
        return "claude"  # default to claude for haiku

    # ── Message normalisation ─────────────────────────────────────────────────

    def _normalise(self, messages) -> Tuple[str, List[dict]]:
        """Convert langchain-style messages → (system_prompt, chat_messages)."""
        system = ""
        chat   = []
        for m in messages:
            if isinstance(m, tuple):
                role, content = m[0], m[1]
            elif hasattr(m, "type"):          # langchain BaseMessage
                role    = "user" if m.type == "human" else m.type
                content = m.content
            elif isinstance(m, dict):
                role, content = m.get("role", "user"), m.get("content", "")
            else:
                role, content = "user", str(m)

            if role == "system":
                system = content
            else:
                role = "user" if role in ("human",) else role
                chat.append({"role": role, "content": content})
        return system, chat

    # ── Payload builders ──────────────────────────────────────────────────────

    def _build_payload(self, messages) -> dict:
        system, chat = self._normalise(messages)

        if self._family == "llama":
            prompt = ""
            if system:
                prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>"
            else:
                prompt += "<|begin_of_text|>"
            for m in chat:
                prompt += f"<|start_header_id|>{m['role']}<|end_header_id|>\n{m['content']}<|eot_id|>"
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
            return {"prompt": prompt, "max_gen_len": self.max_tokens, "temperature": self.temperature}

        elif self._family == "nova":
            payload = {
                "messages": chat,
                "inferenceConfig": {"maxTokens": self.max_tokens, "temperature": self.temperature},
            }
            if system:
                payload["system"] = [{"text": system}]
            return payload

        elif self._family == "claude":
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": chat,
            }
            if system:
                payload["system"] = system
            return payload

        else:  # titan
            full = (system + "\n" if system else "") + "\n".join(m["content"] for m in chat)
            return {"inputText": full, "textGenerationConfig": {"maxTokenCount": self.max_tokens, "temperature": self.temperature}}

    # ── Response extractors ───────────────────────────────────────────────────

    def _extract_text(self, data: dict) -> str:
        if self._family == "llama":
            return data.get("generation", "")
        elif self._family == "nova":
            try:
                return data["output"]["message"]["content"][0]["text"]
            except (KeyError, IndexError):
                return str(data)
        elif self._family == "claude":
            try:
                return data["content"][0]["text"]
            except (KeyError, IndexError):
                return str(data)
        else:  # titan
            try:
                return data["results"][0]["outputText"]
            except (KeyError, IndexError):
                return str(data)

    # ── Public API ────────────────────────────────────────────────────────────

    def invoke(self, messages) -> _BedrockResponse:
        """Synchronous call — same as langchain .invoke()"""
        payload = self._build_payload(messages)
        resp = self._client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        data = json.loads(resp["body"].read())
        text = self._extract_text(data)
        logger.debug(f"[BedrockLLM] {self.model_id} → {len(text)} chars")
        return _BedrockResponse(text)

    async def ainvoke(self, messages) -> _BedrockResponse:
        """Async call — same as langchain .ainvoke()"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, messages)
