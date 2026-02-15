"""LLM provider abstraction for FormBridge.

This module provides a protocol-based abstraction for LLM providers,
allowing FormBridge to work with OpenAI, Anthropic, and any OpenAI-compatible API
using only HTTP calls (no SDK dependencies).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx


class LLMProvider(Protocol):
    """Protocol for LLM providers.

    All LLM calls go through this interface to ensure compatibility
    across different providers (OpenAI, Anthropic, local models).
    """

    def complete(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Send a completion request to the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            schema: Optional JSON schema to enforce structured output
            temperature: Sampling temperature (0.0 = deterministic)

        Returns:
            Response dict with 'content' key containing the response
        """
        ...


class LLMError(Exception):
    """Base error for LLM operations."""
    pass


class LLMConfigError(LLMError):
    """Error for LLM configuration issues."""
    pass


class LLMAPIError(LLMError):
    """Error for LLM API failures."""

    def __init__(self, message: str, status_code: int | None = None, response_body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    provider: str
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 120.0

    @property
    def effective_model(self) -> str:
        """Get the effective model name (with defaults)."""
        if self.model:
            return self.model
        if self.provider == "openai":
            return "gpt-4o-mini"
        if self.provider == "anthropic":
            return "claude-3-5-sonnet-20241022"
        return "unknown"


class BaseLLMProvider(ABC):
    """Base class for LLM providers with common functionality."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        """Initialize the provider.

        Args:
            api_key: API key for the provider
            model: Model name to use
            base_url: Custom base URL for the API
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            headers = self._get_headers()
            self._client = httpx.Client(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    @abstractmethod
    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for requests."""
        pass

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> BaseLLMProvider:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible LLM provider.

    Works with:
    - OpenAI API (gpt-4o, gpt-4o-mini, etc.)
    - Ollama (with /v1 endpoints)
    - LM Studio
    - Any OpenAI-compatible API
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        """Initialize OpenAI-compatible provider.

        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            model: Model name (defaults to gpt-4o-mini)
            base_url: API base URL (defaults to OpenAI's API)
            timeout: Request timeout
        """
        # Resolve API key
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key and not base_url:
            # Local models often don't need API keys
            pass

        super().__init__(
            api_key=resolved_key,
            model=model or self.DEFAULT_MODEL,
            base_url=base_url or self.DEFAULT_BASE_URL,
            timeout=timeout,
        )

    def _get_headers(self) -> dict[str, str]:
        """Get headers for OpenAI API requests."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def complete(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Send completion request to OpenAI-compatible API.

        Args:
            messages: List of message dicts
            schema: Optional JSON schema for structured output
            temperature: Sampling temperature

        Returns:
            Response dict with 'content' containing the response
        """
        if not self.model:
            raise LLMConfigError("Model name is required")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        # Add structured output if schema provided
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": schema,
                    "strict": True,
                },
            }

        try:
            response = self.client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise LLMAPIError(
                f"OpenAI API error: {e.response.status_code}",
                status_code=e.response.status_code,
                response_body=e.response.text,
            ) from e
        except httpx.RequestError as e:
            raise LLMAPIError(f"Request failed: {e}") from e

        data = response.json()

        # Extract content from response
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMAPIError(f"Unexpected response format: {data}") from e

        # Parse JSON if structured output was requested
        if schema and isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError as e:
                raise LLMAPIError(f"Failed to parse JSON response: {content}") from e

        return {
            "content": content,
            "model": data.get("model"),
            "usage": data.get("usage"),
        }


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider."""

    DEFAULT_BASE_URL = "https://api.anthropic.com"
    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        """Initialize Anthropic provider.

        Args:
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model name (defaults to claude-3-5-sonnet)
            base_url: API base URL
            timeout: Request timeout
        """
        # Resolve API key
        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise LLMConfigError(
                "Anthropic API key is required. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key."
            )

        super().__init__(
            api_key=resolved_key,
            model=model or self.DEFAULT_MODEL,
            base_url=base_url or self.DEFAULT_BASE_URL,
            timeout=timeout,
        )

    def _get_headers(self) -> dict[str, str]:
        """Get headers for Anthropic API requests."""
        return {
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key or "",
            "anthropic-version": self.API_VERSION,
        }

    def complete(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Send completion request to Anthropic API.

        Args:
            messages: List of message dicts
            schema: Optional JSON schema for structured output (beta feature)
            temperature: Sampling temperature

        Returns:
            Response dict with 'content' containing the response
        """
        if not self.model:
            raise LLMConfigError("Model name is required")

        # Convert OpenAI-style messages to Anthropic format
        system_message = None
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_message = content
            else:
                anthropic_messages.append({
                    "role": role,
                    "content": content,
                })

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": 4096,
        }

        if system_message:
            payload["system"] = system_message

        # Add structured output via tools if schema provided
        if schema:
            # Use tools beta feature for structured output
            tool_name = "structured_output"
            payload["tools"] = [{
                "name": tool_name,
                "description": "Structured output format",
                "input_schema": schema,
            }]
            payload["tool_choice"] = {"type": "tool", "name": tool_name}

        try:
            response = self.client.post("/v1/messages", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise LLMAPIError(
                f"Anthropic API error: {e.response.status_code}",
                status_code=e.response.status_code,
                response_body=e.response.text,
            ) from e
        except httpx.RequestError as e:
            raise LLMAPIError(f"Request failed: {e}") from e

        data = response.json()

        # Extract content from response
        try:
            content_blocks = data["content"]

            # Handle tool use for structured output
            if schema:
                for block in content_blocks:
                    if block.get("type") == "tool_use":
                        content = block.get("input", {})
                        break
                else:
                    raise LLMAPIError(f"Expected tool_use in response: {data}")
            else:
                # Concatenate text blocks
                content = "".join(
                    block.get("text", "")
                    for block in content_blocks
                    if block.get("type") == "text"
                )
        except (KeyError, IndexError) as e:
            raise LLMAPIError(f"Unexpected response format: {data}") from e

        return {
            "content": content,
            "model": data.get("model"),
            "usage": {
                "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
            },
        }


def create_provider(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMProvider:
    """Create an LLM provider based on configuration.

    Configuration priority:
    1. Explicit parameters
    2. Environment variables (FORMBRIDGE_PROVIDER, FORMBRIDGE_MODEL, etc.)
    3. Defaults (OpenAI with gpt-4o-mini)

    Args:
        provider: Provider name ('openai', 'anthropic', 'local')
        model: Model name
        api_key: API key
        base_url: Custom base URL

    Returns:
        Configured LLMProvider instance

    Raises:
        LLMConfigError: If configuration is invalid
    """
    # Resolve provider from env or default
    resolved_provider = (provider or os.getenv("FORMBRIDGE_PROVIDER", "openai")).lower()
    resolved_model = model or os.getenv("FORMBRIDGE_MODEL")
    resolved_api_key = api_key or os.getenv("FORMBRIDGE_API_KEY")
    resolved_base_url = base_url or os.getenv("FORMBRIDGE_API_BASE")

    if resolved_provider in ("openai", "local"):
        # For local models, use OpenAI-compatible interface
        if resolved_provider == "local" and not resolved_base_url:
            resolved_base_url = "http://localhost:11434/v1"  # Default Ollama endpoint

        return OpenAIProvider(
            api_key=resolved_api_key,
            model=resolved_model,
            base_url=resolved_base_url,
        )

    elif resolved_provider == "anthropic":
        return AnthropicProvider(
            api_key=resolved_api_key or os.getenv("ANTHROPIC_API_KEY"),
            model=resolved_model,
            base_url=resolved_base_url,
        )

    else:
        raise LLMConfigError(f"Unknown provider: {resolved_provider}")


def load_config(
    provider: str | None = None,
    model: str | None = None,
    config_path: str | None = None,
) -> LLMConfig:
    """Load LLM configuration from environment and optional config file.

    Args:
        provider: Optional provider override
        model: Optional model override
        config_path: Optional path to config file

    Returns:
        LLMConfig instance
    """
    # Start with environment/config file defaults
    resolved_provider = provider or os.getenv("FORMBRIDGE_PROVIDER", "openai")
    resolved_model = model or os.getenv("FORMBRIDGE_MODEL")
    api_key = os.getenv("FORMBRIDGE_API_KEY")
    base_url = os.getenv("FORMBRIDGE_API_BASE")

    # Try to load from config file if provided
    if config_path is None:
        config_path = "formbridge.toml"

    if Path(config_path).exists():
        try:
            import tomllib
            with open(config_path, "rb") as f:
                file_config = tomllib.load(f)

            llm_config = file_config.get("llm", {})
            resolved_provider = provider or llm_config.get("provider", resolved_provider)
            resolved_model = model or llm_config.get("model", resolved_model)
            base_url = base_url or llm_config.get("base_url")

            # Handle API key from env var reference
            api_key_env = llm_config.get("api_key_env")
            if api_key_env:
                api_key = os.getenv(api_key_env)

            # Handle local config section
            if resolved_provider == "local" or "local" in llm_config:
                local_config = llm_config.get("local", {})
                base_url = base_url or local_config.get("base_url")
                resolved_model = resolved_model or local_config.get("model")
        except Exception:
            # Ignore config file errors, use env defaults
            pass

    # Provider-specific API key resolution
    if resolved_provider == "anthropic" and not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif resolved_provider == "openai" and not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    return LLMConfig(
        provider=resolved_provider,
        model=resolved_model,
        api_key=api_key,
        base_url=base_url,
    )


def get_provider_from_config(config_path: str | None = None) -> LLMProvider:
    """Create provider from formbridge.toml config file.

    Args:
        config_path: Path to config file (defaults to formbridge.toml in cwd)

    Returns:
        Configured LLMProvider
    """
    config = load_config(config_path=config_path)

    return create_provider(
        provider=config.provider,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
    )
