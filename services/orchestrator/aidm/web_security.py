"""Shared HTTP-layer security plumbing for the WebRTC and text servers.

Three concerns live here:

1. **CORS** — env-driven origin allowlist, falls back to dev defaults so
   local development keeps working.
2. **Rate limiting** — per-IP via SlowAPI, in-memory backend (single
   process, no Redis needed at our scale).
3. **Cloudflare Access JWT verification** — defense in depth behind the
   CF Access gate. When ``CF_ACCESS_TEAM_DOMAIN`` and ``CF_ACCESS_AUD``
   are set, every request must carry a valid ``Cf-Access-Jwt-Assertion``
   header. ``CF_ACCESS_DISABLED=1`` skips verification for local dev.
4. **Daily budget guard** — raises ``HTTPException(503)`` when today's
   spend exceeds ``AIDM_DAILY_USD_CAP``. The HTTP layer calls this
   before launching a new pipeline; the per-session in-pipeline guard
   handles intra-session caps.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import jwt
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from aidm.budget import BudgetExceededError, DailyBudgetLedger
from aidm.config import Config

_DEV_ORIGINS: tuple[str, ...] = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
)

_RATE_LIMIT_MESSAGE_HU = (
    "Túl sok kérés egyetlen forrásból — pihenj egy percet, kapitány, aztán próbáld újra."
)
_BUDGET_MESSAGE_HU = "Az archívum mai költségkerete elfogyott. A krónika holnap folytatódik."
_AUTH_MESSAGE_HU = "Hozzáférés megtagadva — érvénytelen vagy hiányzó belépési jegy."


def configure_cors(app: FastAPI, config: Config) -> None:
    """Wire CORS middleware from config.allowed_origins.

    Dev posture (``CF_ACCESS_DISABLED=1``) falls back to localhost origins
    so ``make dev`` works zero-config. Any other posture demands an
    explicit ``AIDM_ALLOWED_ORIGINS`` — refusing to start is the only way
    to keep a forgotten env var from silently exposing the API to
    arbitrary browser origins in prod.
    """
    if config.allowed_origins:
        origins = list(config.allowed_origins)
    elif config.cf_access_disabled:
        origins = list(_DEV_ORIGINS)
    else:
        raise RuntimeError(
            "AIDM_ALLOWED_ORIGINS is empty but CF_ACCESS_DISABLED is not set. "
            "Refusing to start with the dev-origin fallback in a production "
            "posture — set AIDM_ALLOWED_ORIGINS to the comma-separated list "
            "of browser origins, or set CF_ACCESS_DISABLED=1 for local dev."
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "PUT", "OPTIONS"],
        allow_headers=["*"],
        # Cookies aren't used in our flow; keeping False means a wildcard
        # origin would still be safe, but we don't use wildcards anyway.
        allow_credentials=False,
    )


def build_limiter() -> Limiter:
    """In-memory per-IP rate limiter. Apps register handlers via decorators."""
    return Limiter(key_func=get_remote_address)


def install_rate_limit_handler(app: FastAPI, limiter: Limiter) -> None:
    """Wire SlowAPI exception handler and stash the limiter on app.state.

    Routes apply limits via ``@limiter.limit("5/minute")`` decorators.
    """
    app.state.limiter = limiter

    async def _handler(request: Request, exc: Exception) -> JSONResponse:
        # Narrow type: SlowAPI passes RateLimitExceeded here.
        del request
        if not isinstance(exc, RateLimitExceeded):
            raise exc
        return JSONResponse(
            status_code=429,
            content={"detail": _RATE_LIMIT_MESSAGE_HU},
            headers={"Retry-After": "60"},
        )

    app.add_exception_handler(RateLimitExceeded, _handler)


# ----------------------------- CF Access JWT ---------------------------------

# Cache JWKS for an hour. Cloudflare rotates keys but old keys remain valid
# for a transition window, so a stale cache is safe within the TTL.
_JWKS_TTL_SECONDS = 3600


class _JWKSCache:
    """Lazy fetcher for the Cloudflare Access JWKS document."""

    def __init__(self, team_domain: str) -> None:
        self._url = f"https://{team_domain}/cdn-cgi/access/certs"
        self._client: jwt.PyJWKClient | None = None
        self._fetched_at: float = 0.0

    def get(self) -> jwt.PyJWKClient:
        now = time.monotonic()
        if self._client is None or now - self._fetched_at > _JWKS_TTL_SECONDS:
            self._client = jwt.PyJWKClient(
                self._url, cache_jwk_set=True, lifespan=_JWKS_TTL_SECONDS
            )
            self._fetched_at = now
        return self._client


def make_cf_access_verifier(config: Config) -> Callable[[Request], None]:
    """Return a FastAPI dependency that verifies Cloudflare Access JWTs.

    Behavior:
    * If ``CF_ACCESS_DISABLED=1`` (local dev), the dependency is a no-op.
    * If team_domain/AUD are unset and DISABLED is not set: refuse to
      build the app. Open-by-default on a misconfigured prod is the kind
      of footgun we never want — fail loud at boot instead.
    * Otherwise: extract ``Cf-Access-Jwt-Assertion`` header, decode with
      JWKS-fetched RS256 key, verify ``aud``. On failure → 401.
    """
    if config.cf_access_disabled:
        logger.warning(
            "CF Access JWT verification DISABLED (CF_ACCESS_DISABLED=1). "
            "Never run prod with this set."
        )

        def _disabled(_request: Request) -> None:
            return None

        return _disabled

    if not config.cf_access_team_domain or not config.cf_access_aud:
        raise RuntimeError(
            "CF Access not configured: CF_ACCESS_TEAM_DOMAIN and CF_ACCESS_AUD "
            "must both be set when CF_ACCESS_DISABLED is not 1. Refusing to "
            "start in an open-by-default posture."
        )

    jwks = _JWKSCache(config.cf_access_team_domain)
    expected_aud = config.cf_access_aud
    issuer = f"https://{config.cf_access_team_domain}"

    def _verify(request: Request) -> None:
        token = request.headers.get("cf-access-jwt-assertion") or request.headers.get(
            "Cf-Access-Jwt-Assertion"
        )
        if not token:
            raise HTTPException(status_code=401, detail=_AUTH_MESSAGE_HU)
        try:
            signing_key = jwks.get().get_signing_key_from_jwt(token).key
            jwt.decode(
                token,
                signing_key,
                algorithms=["RS256"],
                audience=expected_aud,
                issuer=issuer,
            )
        except jwt.PyJWTError as exc:
            logger.warning("CF Access JWT verification failed: {}", exc)
            raise HTTPException(status_code=401, detail=_AUTH_MESSAGE_HU) from exc

    return _verify


# ----------------------------- Daily budget ----------------------------------


def enforce_daily_budget(ledger: DailyBudgetLedger | None, cap_usd: float) -> None:
    """Raise HTTPException(503) when the daily cap is reached.

    Called at session-creation endpoints (POST /api/offer, POST /api/text/session).
    A ``None`` ledger means budget tracking is disabled (cap <= 0 or non-public mode).
    """
    if ledger is None or cap_usd <= 0:
        return
    try:
        ledger.check_cap(cap_usd)
    except BudgetExceededError as exc:
        raise HTTPException(status_code=503, detail=_BUDGET_MESSAGE_HU) from exc
