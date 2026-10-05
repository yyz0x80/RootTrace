"""Upstream LLM failure taxonomy and retry-delay helpers.

RootTrace retries LLM failures in two layers, and both layers share this
module:

- the Provider uses it to decide which upstream failures deserve another
  in-call attempt and how long to wait for it;
- the queue layer uses it to decide whether a failed RCA job may be retried by
  RQ, and to withdraw the RQ retry budget for deterministic failures.

Keeping the taxonomy in one place guarantees that both layers classify the
same exception identically.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAIError,
    RateLimitError,
)

BACKOFF_BASE_SECONDS = 1.0
BACKOFF_CAP_SECONDS = 8.0
MAX_RETRY_AFTER_SECONDS = 30.0
MAX_TOTAL_RETRY_WAIT_SECONDS = 60.0

# Payload markers that identify exhausted quota/billing instead of a burst rate
# limit. Quota exhaustion is deterministic, so retrying only burns budget.
_QUOTA_EXHAUSTED_MARKERS = (
    "insufficient_quota",
    "exceeded your current quota",
    "billing",
)


def retry_after_seconds(error: BaseException) -> float | None:
    """Return the server-requested retry delay in seconds, when advertised.

    OpenAI-compatible providers may include a ``Retry-After`` header (delta
    seconds or an HTTP date) on 429 or 5xx responses. Honoring it avoids
    retrying too early; a missing or unparsable header returns ``None`` so the
    caller falls back to jittered exponential backoff. Exceptions without
    response headers also return ``None``.
    """
    response = getattr(error, "response", None)
    headers = getattr(error, "headers", None) or getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("Retry-After") or headers.get("retry-after")
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if retry_at.tzinfo is None:
        now = datetime.now(UTC).replace(tzinfo=None)
    else:
        now = datetime.now(UTC)
    return max((retry_at - now).total_seconds(), 0.0)


def retry_delay_seconds(
    attempt: int,
    *,
    retry_after_seconds: float | None = None,
    rng: random.Random | None = None,
) -> float:
    """Compute a bounded, jittered delay before the next transient retry.

    A server-provided ``Retry-After`` value takes precedence and is capped to
    keep total retry time bounded. Otherwise the delay uses capped exponential
    backoff with full jitter so concurrent specialists do not retry in lockstep
    and re-trigger the same limit together.
    """
    if retry_after_seconds is not None:
        return min(max(retry_after_seconds, 0.0), MAX_RETRY_AFTER_SECONDS)
    backoff = min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * (2**attempt))
    rng = rng or random
    return rng.uniform(0.0, backoff)


def llm_retry_delay_seconds(
    error: BaseException,
    attempt: int,
    rng: random.Random | None = None,
) -> float:
    """Return the bounded wait before retrying one transient LLM failure.

    The wait honors an advertised ``Retry-After`` header when the failure
    carries one and falls back to jittered exponential backoff otherwise.
    """
    return retry_delay_seconds(
        attempt,
        retry_after_seconds=retry_after_seconds(error),
        rng=rng,
    )


def is_quota_exhausted(error: BaseException) -> bool:
    """Return whether the error reports exhausted quota/billing.

    Providers surface exhausted quota either as a 429 ``RateLimitError`` with an
    ``insufficient_quota`` payload or as a 4xx billing error. Both are
    deterministic and must not be retried.
    """
    if not isinstance(error, APIStatusError):
        return False
    text = " ".join(
        str(part)
        for part in (
            getattr(error, "code", None),
            getattr(error, "type", None),
            getattr(error, "body", None),
            str(error),
        )
        if part
    ).lower()
    return any(marker in text for marker in _QUOTA_EXHAUSTED_MARKERS)


def is_llm_error(error: BaseException) -> bool:
    """Return whether the failure came from the LLM provider layer."""
    return isinstance(error, OpenAIError)


def is_retryable_llm_error(error: BaseException) -> bool:
    """Return whether an LLM failure is transient and worth another attempt.

    Rate limits (429 without quota exhaustion), connection failures, request
    timeouts, and server-side (5xx) errors may succeed on a later attempt.
    Client errors (4xx), authentication failures, and exhausted quota/billing
    are deterministic and must fail fast instead of burning retry budget.
    """
    if not isinstance(error, OpenAIError):
        return False
    if isinstance(error, RateLimitError):
        return not is_quota_exhausted(error)
    if isinstance(error, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code >= 500
    return False


def is_non_retryable_llm_error(error: BaseException) -> bool:
    """Return whether an LLM failure is deterministic and must not be retried."""
    return is_llm_error(error) and not is_retryable_llm_error(error)
