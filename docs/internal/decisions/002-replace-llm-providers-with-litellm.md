# ADR 002: Replace Hand-Rolled LLM Providers with litellm

*Date:* 2026-04-05
*Status:* proposed

## Context

FormBridge ships a hand-rolled LLM abstraction in `llm.py` (~540 lines) consisting of:

- `LLMProvider` protocol with `complete(messages, schema, temperature)`
- `OpenAIProvider` -- raw `httpx` calls to `/chat/completions`
- `AnthropicProvider` -- raw `httpx` calls to `/v1/messages` with manual format conversion
- `create_provider()` factory with env var / TOML config resolution
- `load_config()` / `get_provider_from_config()` for config file support

This covers two providers and text-only completions with optional JSON schema structured output. Three call sites consume it:

1. `mapper.py:284` -- `self.provider.complete(messages, schema=output_schema)` for field mapping
2. `parser.py:301` -- same pattern for instruction parsing
3. `parser.py:437` -- same pattern for calculation extraction

All three call sites pass the same shape: OpenAI-style `messages` list + optional JSON `schema`. The return shape consumed is `{"content": <str|dict>, "model": ..., "usage": ...}`.

### Problems with the current approach

1. **No multimodal/vision support.** ADR 001 requires sending annotated PDF page images to a vision model. `OpenAIProvider.complete()` can technically pass through OpenAI-format image content blocks, but `AnthropicProvider.complete()` does manual message conversion at line 331-341 that only handles string `content` -- image blocks would be silently dropped.

2. **Provider-specific format translation is fragile.** The Anthropic provider hand-converts messages (extracts system, restructures content), hand-handles structured output via tool_use, and hand-parses response blocks. This broke in practice with z.ai (an OpenAI-compatible endpoint) due to base URL path-joining issues.

3. **Only two providers.** Google Gemini, Vertex AI, Azure OpenAI, Ollama multimodal models, Mistral, etc. are all unsupported. Each new provider would require another ~100-line provider class.

4. **No built-in retry/fallback.** If the primary model rate-limits or errors, FormBridge fails immediately. litellm provides configurable retries and model fallbacks.

5. **No async.** The current providers are synchronous (`httpx.Client`). Vision refinement (ADR 001) will be I/O-bound and benefits from async. litellm supports both sync and async natively.

### litellm capabilities

litellm provides a single `completion()` function that:

- Accepts OpenAI-format messages (including `image_url` content blocks for vision)
- Routes to 100+ providers via model name prefix (`openai/`, `anthropic/`, `gemini/`, `vertex_ai/`, `ollama/`, etc.)
- Handles provider-specific message format translation internally
- Supports base URL override per call (`api_base`) or via env vars
- Supports JSON schema structured output
- Provides built-in retry with fallback models
- Has both sync (`completion`) and async (`acompletion`) APIs
- Handles base64-encoded images inline (`data:image/png;base64,...`)

## Decision

Replace `OpenAIProvider`, `AnthropicProvider`, and `BaseLLMProvider` with a single litellm-backed implementation. Keep `LLMProvider` protocol, `LLMConfig`, and the config resolution functions as the public API -- internal consumers don't change.

### What changes

**`llm.py` shrinks** from ~540 lines to ~150 lines:

- `LLMProvider` protocol stays (unchanged)
- `LLMConfig` stays (unchanged)
- `LLMError`, `LLMConfigError`, `LLMAPIError` stay (map litellm exceptions)
- `OpenAIProvider` and `AnthropicProvider` are replaced by a single `LiteLLMProvider`
- `create_provider()` returns `LiteLLMProvider` regardless of provider string
- `load_config()` and `get_provider_from_config()` stay (they just produce `LLMConfig`)
- `httpx` dependency moves from core to optional (litellm brings its own HTTP)

**`LiteLLMProvider` implementation:**

```python
class LiteLLMProvider:
    """LLM provider backed by litellm.

    Handles all providers via litellm's unified completion API.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._model = self._resolve_model_name(config)

    def _resolve_model_name(self, config: LLMConfig) -> str:
        """Convert FormBridge provider+model to litellm model string.

        FormBridge config uses 'openai', 'anthropic', etc.
        litellm uses prefixed names like 'openai/gpt-4o', 'anthropic/claude-sonnet-4-20250514'.
        """
        if config.model and "/" in config.model:
            return config.model  # Already a litellm-prefixed name

        provider_prefixes = {
            "openai": "openai",
            "anthropic": "anthropic",
            "local": "ollama",
        }
        prefix = provider_prefixes.get(config.provider, config.provider)
        model = config.model or self._default_model(config.provider)
        return f"{prefix}/{model}"

    def complete(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
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
        except litellm.APIError as e:
            raise LLMAPIError(str(e), status_code=e.status_code) from e
        except litellm.AuthenticationError as e:
            raise LLMConfigError(str(e)) from e
        except Exception as e:
            raise LLMAPIError(f"litellm error: {e}") from e

        choice = response.choices[0]
        content = choice.message.content

        if schema and isinstance(content, str):
            content = json.loads(content)

        return {
            "content": content,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        }
```

### What stays the same

- `LLMProvider.complete(messages, schema, temperature) -> dict` -- the protocol interface is unchanged
- `LLMConfig(provider, model, api_key, base_url, timeout)` -- config shape unchanged
- `create_provider()`, `load_config()`, `get_provider_from_config()` -- call signatures unchanged
- All three consumer call sites (`mapper.py:284`, `parser.py:301`, `parser.py:437`) -- zero changes
- `__init__.py` re-exports -- unchanged
- `formbridge.toml` config format -- unchanged

### Dependency change

```toml
# pyproject.toml

dependencies = [
    "click>=8.1.0",
    "litellm>=1.50.0",       # replaces httpx for LLM calls
    "pydantic>=2.0.0",
    "pdfplumber>=0.10.0",
    "pikepdf>=8.0.0",
    "rich>=13.0.0",
    "reportlab>=4.0.0",
]

# httpx moves to optional (only needed if other code uses it directly)
```

litellm becomes a core dependency (not optional) because `mapper.py` and `parser.py` already require LLM calls for their primary function.

### Model name migration

Existing `formbridge.toml` configs use provider-specific model names:

```toml
# Before (still works -- create_provider maps it)
[llm]
provider = "openai"
model = "gpt-4o-mini"

# After (litellm-prefixed names also accepted)
[llm]
model = "openai/gpt-4o-mini"
```

`LiteLLMProvider._resolve_model_name()` handles both formats. No config migration required.

## Alternatives Considered

1. **Keep hand-rolled providers, add a VisionProvider separately** -- Add a third provider class specifically for vision calls. Rejected: duplicates the format translation problem for image content. The Anthropic message converter would need a parallel path for multimodal content blocks. litellm already handles this.

2. **Keep hand-rolled providers, patch AnthropicProvider for vision** -- Fix `AnthropicProvider.complete()` to handle image content blocks and add an `httpx`-based vision call method. Rejected: this is building a worse litellm. Every new provider still needs its own class.

3. **Replace with litellm (chosen)** -- Single provider class, all format translation delegated to litellm. Vision works by passing standard OpenAI-format image content blocks. New providers work by changing the model name string. Retries and fallbacks are free.

4. **Replace with instructor** -- instructor wraps litellm with Pydantic model validation. Rejected: FormBridge already has its own Pydantic models for structured output. Adding instructor adds a second validation layer. litellm's `response_format` with JSON schema is sufficient.

5. **Wrap litellm behind a thin shim only for vision, keep existing providers for text** -- Use litellm only in the new vision refinement code. Rejected: this means maintaining two LLM call paths (hand-rolled for text, litellm for vision). The hand-rolled path has known bugs (z.ai base URL) that would remain unfixed.

## Consequences

- **Pros:**
  - Vision support (ADR 001) works by passing `image_url` content blocks -- no provider-specific image handling code
  - All providers supported immediately (Gemini, Vertex, Ollama, Azure, Mistral, etc.)
  - Fixes the z.ai base URL issue for free (litellm handles URL construction per provider)
  - Built-in retry with exponential backoff
  - Async support (`litellm.acompletion`) available when needed for parallel vision calls
  - `llm.py` shrinks by ~390 lines -- less code to maintain
  - No changes to consumer call sites or public API

- **Cons:**
  - litellm is a large dependency (~many MB, many transitive deps). FormBridge currently has a small footprint.
  - Less control over the exact HTTP request/response cycle. Debugging provider issues means reading litellm source, not FormBridge source.
  - litellm version churn. The project is actively developed and occasionally breaks backward compatibility in model name routing.
  - FormBridge users who previously installed without any LLM provider (using only the scanner) now get litellm installed. This is acceptable since `mapper` and `parser` already require an LLM.

## Technical Details

### litellm vision message format

For ADR 001's vision refinement, the call would be:

```python
import base64
import litellm

b64_image = base64.b64encode(annotated_page_png_bytes).decode()

response = litellm.completion(
    model="openai/gpt-4o",  # or "anthropic/claude-sonnet-4-20250514", etc.
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Identify the printed label for each numbered field..."},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{b64_image}"
            }}
        ]
    }],
    response_format={"type": "json_object"},
    temperature=0.0,
)
```

litellm translates the `image_url` content block to each provider's native format:

- OpenAI: passes through as-is
- Anthropic: converts to `{"type": "image", "source": {"type": "base64", ...}}`
- Gemini: converts to `{"inlineData": {"mimeType": "image/png", "data": ...}}`

No provider-specific code needed in FormBridge.

### Migration path

1. Add `litellm>=1.50.0` to core dependencies
2. Implement `LiteLLMProvider` alongside existing providers (no deletion yet)
3. Switch `create_provider()` to return `LiteLLMProvider`
4. Run existing test suite -- `LLMProvider.complete()` interface is unchanged, so mocks/stubs still work
5. Remove `OpenAIProvider`, `AnthropicProvider`, `BaseLLMProvider`
6. Remove `httpx` from core deps if no other code uses it directly

### Files changed

| File | Change |
|------|--------|
| `pyproject.toml` | Add `litellm>=1.50.0` to deps, optionally remove `httpx` |
| `src/formbridge/llm.py` | Replace 3 provider classes with `LiteLLMProvider`, keep config/protocol |
| `tests/test_llm.py` | Update tests to mock `litellm.completion` instead of `httpx.Client` |
| `src/formbridge/mapper.py` | No changes (consumes `LLMProvider` protocol) |
| `src/formbridge/parser.py` | No changes (consumes `LLMProvider` protocol) |
| `src/formbridge/cli.py` | No changes |
| `src/formbridge/mcp_server.py` | No changes |
| `src/formbridge/templates.py` | No changes |

## Supersedes / Dependencies

- enables: ADR 001 (vision-augmented label extraction) -- litellm's multimodal support is the implementation path
- supersedes: the hand-rolled `OpenAIProvider` and `AnthropicProvider` in `llm.py`
- related: upstream `nilsyai/formbridge` may benefit from this change; consider contributing
