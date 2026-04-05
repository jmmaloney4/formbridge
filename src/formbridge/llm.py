"""LLM provider abstraction for FormBridge.

This module provides a protocol-based abstraction for LLM providers,
powered by litellm for unified access to 100+ providers (OpenAI, Anthropic,
Gemini, Ollama, etc.) including multimodal/vision support.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


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


# Provider name -> litellm prefix mapping
_PROVIDER_PREFIXES: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "local": "ollama",
}

# Provider name -> default model (without prefix)
_PROVIDER_DEFAULTS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-20241022",
    "local": "llama3.1",
}


def _resolve_model_name(provider: str, model: str | None) -> str:
    """Convert FormBridge provider+model to litellm model string.

    Handles both old-style config ('openai' + 'gpt-4o-mini') and
    litellm-prefixed names ('openai/gpt-4o-mini').
    """
    if model and "/" in model:
        return model  # Already a litellm-prefixed name

    prefix = _PROVIDER_PREFIXES.get(provider, provider)
    resolved = model or _PROVIDER_DEFAULTS.get(provider, "gpt-4o-mini")
    return f"{prefix}/{resolved}"


class LiteLLMProvider:
    """LLM provider backed by litellm.

    Handles all providers via litellm's unified completion API,
    including multimodal/vision content blocks.
    """

    def __init__(
        self,
        config: LLMConfig,
    ) -> None:
        self.config = config
        self._model = _resolve_model_name(config.provider, config.model)

    def complete(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Send completion request via litellm.

        Args:
            messages: List of message dicts (OpenAI format, supports image_url content)
            schema: Optional JSON schema for structured output
            temperature: Sampling temperature

        Returns:
            Response dict with 'content', 'model', and 'usage' keys

        Raises:
            LLMConfigError: If configuration is invalid
            LLMAPIError: If the API call fails
        """
        import litellm

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }

        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.base_url:
            kwargs["api_base"] = self.config.base_url

        # Add structured output if schema provided
        if schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": schema,
                    "strict": True,
                },
            }

        try:
            response = litellm.completion(**kwargs)
        except litellm.AuthenticationError as e:
            raise LLMConfigError(str(e)) from e
        except litellm.APIError as e:
            raise LLMAPIError(
                str(e),
                status_code=getattr(e, "status_code", None),
            ) from e
        except litellm.BadRequestError as e:
            raise LLMAPIError(
                str(e),
                status_code=getattr(e, "status_code", 400),
            ) from e
        except litellm.RateLimitError as e:
            raise LLMAPIError(
                str(e),
                status_code=getattr(e, "status_code", 429),
            ) from e
        except litellm.Timeout as e:
            raise LLMAPIError(f"Request timed out: {e}") from e
        except Exception as e:
            raise LLMAPIError(f"litellm error: {e}") from e

        choice = response.choices[0]
        content = choice.message.content

        # Parse JSON if structured output was requested
        if schema and isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError as e:
                raise LLMAPIError(f"Failed to parse JSON response: {content}") from e

        usage = response.usage
        return {
            "content": content,
            "model": response.model,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
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

    # For local models, default to Ollama endpoint
    if resolved_provider == "local" and not resolved_base_url:
        resolved_base_url = "http://localhost:11434/v1"

    # Provider-specific API key resolution
    if resolved_provider == "openai" and not resolved_api_key:
        resolved_api_key = os.getenv("OPENAI_API_KEY")
    elif resolved_provider == "anthropic" and not resolved_api_key:
        resolved_api_key = os.getenv("ANTHROPIC_API_KEY")

    config = LLMConfig(
        provider=resolved_provider,
        model=resolved_model,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
    )

    return LiteLLMProvider(config=config)


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
