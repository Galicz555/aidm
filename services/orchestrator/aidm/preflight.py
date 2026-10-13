"""Provider preflight checks.

A bad API key or revoked voice ID otherwise surfaces ~5 seconds into the
first turn — the user has already spoken before they learn the run is
broken. We probe each public-mode provider with a cheap auth-only call
before the voice loop starts so failures show up as a clean startup
error instead of a mid-conversation crash.

The checks deliberately do not transcribe, generate, or synthesize —
just enough to confirm the credentials answer.
"""

from __future__ import annotations

import asyncio

import aiohttp
from loguru import logger

from aidm.config import Config

_TIMEOUT = aiohttp.ClientTimeout(total=5.0)


class PreflightError(RuntimeError):
    """Raised when a provider rejects the configured credentials."""


async def check_providers(config: Config) -> None:
    """Probe every active provider. Raise on auth failure; warn on flaky network."""
    if config.mode != "public":
        return

    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        checks = [
            _check_openrouter(session, config),
            _check_elevenlabs(session, config),
        ]
        if config.stt_provider == "groq":
            checks.append(_check_groq(session, config))
        # ElevenLabs STT shares the ElevenLabs key already covered above.
        await asyncio.gather(*checks)


async def _check_openrouter(session: aiohttp.ClientSession, config: Config) -> None:
    url = "https://openrouter.ai/api/v1/auth/key"
    headers = {"Authorization": f"Bearer {config.openrouter_api_key}"}
    await _probe("openrouter", session, url, headers=headers)


async def _check_elevenlabs(session: aiohttp.ClientSession, config: Config) -> None:
    # GET /voices/{id} confirms both the API key AND that the configured
    # voice still exists on the account — catches deleted voices too.
    url = f"https://api.elevenlabs.io/v1/voices/{config.elevenlabs_voice_id}"
    headers = {"xi-api-key": config.elevenlabs_api_key}
    await _probe("elevenlabs", session, url, headers=headers)


async def _check_groq(session: aiohttp.ClientSession, config: Config) -> None:
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {config.groq_api_key}"}
    await _probe("groq", session, url, headers=headers)


async def _probe(
    name: str,
    session: aiohttp.ClientSession,
    url: str,
    *,
    headers: dict[str, str],
) -> None:
    try:
        async with session.get(url, headers=headers) as resp:
            if resp.status in (401, 403):
                raise PreflightError(
                    f"{name}: credentials rejected (HTTP {resp.status}). "
                    f"Check the API key / voice id in .env."
                )
            if resp.status == 404:
                raise PreflightError(
                    f"{name}: resource not found (HTTP 404) at {url}. "
                    f"Likely a deleted voice id or wrong endpoint."
                )
            if resp.status >= 500:
                logger.warning(
                    "{} preflight: {} returned {} — provider may be degraded, continuing",
                    name,
                    url,
                    resp.status,
                )
                return
            resp.raise_for_status()
            logger.info("{} preflight ok", name)
    except (TimeoutError, aiohttp.ClientConnectionError) as exc:
        logger.warning(
            "{} preflight: network unreachable ({}). Continuing — the voice loop "
            "will surface this if it persists.",
            name,
            exc,
        )
