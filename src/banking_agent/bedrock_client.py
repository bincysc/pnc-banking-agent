"""
Thin client wrapper around the Bedrock Converse API.

Centralizes model invocation, token usage logging, and error handling.
The agent depends on this abstraction rather than calling boto3 directly,
which keeps the agent provider-agnostic and makes Bedrock calls observable
and testable.
"""

import logging
import time
from typing import Any

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from banking_agent.config import get_config

logger = logging.getLogger(__name__)


class BedrockClient:
    """
    Wraps the Bedrock Runtime Converse API.

    Production design notes:
    - Connection pooling and retries are configured in the boto3 client.
    - Every invocation logs token usage and latency for cost observability.
    - Failures are caught and re-raised with structured context for tracing.
    """

    def __init__(self) -> None:
        config = get_config()
        self._model_id = config.bedrock_model_id
        self._temperature = config.bedrock_temperature
        self._max_tokens = config.bedrock_max_tokens

        # Boto config: connection pooling, retry policy, timeouts.
        # Adaptive retries handle throttling more intelligently than legacy.
        # Read timeout is bounded to prevent hung requests from holding threads.
        boto_config = BotoConfig(
            region_name=config.aws_region,
            retries={"max_attempts": 3, "mode": "adaptive"},
            read_timeout=60,
            connect_timeout=5,
            max_pool_connections=10,
        )

        self._client = boto3.client("bedrock-runtime", config=boto_config)

        # Running token totals — useful for development cost visibility.
        # In production these would emit to CloudWatch metrics.
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._invocation_count = 0

    def converse(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, str]] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Invoke the model via the Bedrock Converse API.

        Args:
            messages: Conversation history in Converse schema.
            system: Optional system prompts in Converse schema.
            tools: Optional tool definitions in Converse schema.

        Returns:
            The raw Converse response dictionary, which contains the model's
            output message and any tool-use blocks the model emitted.

        Raises:
            ClientError: If Bedrock rejects the request (auth, throttling,
                         invalid model ID, malformed request).
        """
        request: dict[str, Any] = {
            "modelId": self._model_id,
            "messages": messages,
            "inferenceConfig": {
                "temperature": self._temperature,
                "maxTokens": self._max_tokens,
            },
        }

        if system:
            request["system"] = system

        if tools:
            request["toolConfig"] = {"tools": tools}

        start_time = time.monotonic()
        try:
            response = self._client.converse(**request)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            logger.error(
                "Bedrock converse failed: code=%s model=%s message_count=%d",
                error_code,
                self._model_id,
                len(messages),
            )
            raise

        latency_ms = (time.monotonic() - start_time) * 1000

        # Update running totals and log per-call cost telemetry.
        usage = response.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)

        self._invocation_count += 1
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens

        logger.info(
            "bedrock.converse model=%s input_tokens=%d output_tokens=%d latency_ms=%.0f "
            "session_totals: invocations=%d input=%d output=%d",
            self._model_id,
            input_tokens,
            output_tokens,
            latency_ms,
            self._invocation_count,
            self._total_input_tokens,
            self._total_output_tokens,
        )

        return response

    @property
    def usage_summary(self) -> dict[str, int]:
        """Cumulative token usage for the lifetime of this client instance."""
        return {
            "invocations": self._invocation_count,
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
        }
