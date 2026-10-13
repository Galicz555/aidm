"""OpenRouter LLM — public-mode response generation.

OpenRouter is an OpenAI-compatible API gateway in front of every major
model (Claude, GPT, Gemini, DeepSeek, ...). One account, one key, one
billing surface. We use it instead of the Anthropic console so a single
top-up funds all model experiments.

The default model is Claude Sonnet 4.5 (``anthropic/claude-sonnet-4.5``)
— best-in-class character work, permissive enough for dark fantasy,
fluent Hungarian. Override ``OPENROUTER_MODEL`` in .env to try Sonnet
4.6 if available under your account, or to swap in GPT/DeepSeek/etc.

Tools are attached to the ``LLMContext`` in ``pipeline.py``; the
handlers are registered via ``register_function`` — same API across all
Pipecat LLM services, so ``vault_tools.py`` works unchanged.
"""

from __future__ import annotations

from pipecat.services.openai.llm import OpenAILLMService

from aidm.config import Config
from aidm.persona import persona_for
from aidm.session import SessionContext

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_openrouter_llm(
    config: Config,
    session: SessionContext,
    *,
    phase: str = "play",
    vault_summary: str = "",
) -> OpenAILLMService:
    return OpenAILLMService(
        api_key=config.openrouter_api_key,
        base_url=_OPENROUTER_BASE_URL,
        settings=OpenAILLMService.Settings(
            model=config.openrouter_model,
            system_instruction=persona_for(
                session.locale,
                emotion_tags=config.tts_supports_emotion_tags,
                mode=config.mode,
                campaign_title=session.campaign_title,
                run_title=session.run_title,
                phase=phase,  # type: ignore[arg-type]
                vault_summary=vault_summary,
            ),
            max_tokens=3072,
            # Hybrid reasoners (DeepSeek V4, Kimi K2) intermittently answer
            # with reasoning-channel tokens and EMPTY content — the turn
            # never produces text and hangs the session. A realtime voice
            # DM never wants thinking latency anyway. OpenRouter drops the
            # param for models without reasoning support.
            extra={"extra_body": {"reasoning": {"enabled": False}}},
        ),
    )
