"""
masis.infra.circuit_breaker
============================
Three-state circuit breaker for MASIS agent LLM calls (ENG-12 / MF-SAFE-02).

State machine
-------------
CLOSED  → normal operation; failure_count tracked on each failure.
OPEN    → after `failure_threshold` consecutive failures; all calls are
          immediately rejected and routed to the fallback function.
          A background timer (`recovery_timeout` seconds) is started.
HALF_OPEN → after the recovery timer fires; exactly ONE probe call is allowed
           through.
           - Probe succeeds → CLOSED (failure_count reset)
           - Probe fails   → OPEN  (timer restarted)

The implementation is fully async-native (``async def call(...)``).  A
synchronous convenience wrapper ``call_sync(...)`` is provided for tests and
scripts that run outside an event loop.

Thread-safety
-------------
State transitions use ``asyncio.Lock`` so that concurrent probe calls do not
both enter HALF_OPEN simultaneously.

Usage
-----
::

    breaker = CircuitBreaker(name="researcher", failure_threshold=4,
                             recovery_timeout=60.0)

    async def primary_fn(query):
        return await researcher_llm(query)

    async def fallback_fn(query):
        return await researcher_llm_gpt4o(query)

    result = await breaker.call(primary_fn, query, fallback_func=fallback_fn)

MF-IDs
------
MF-SAFE-02  3-state circuit breaker (CLOSED / OPEN / HALF_OPEN)
MF-SAFE-03  Model fallback chains — breaker.call() routes to fallback when OPEN
MF-SAFE-07  Graceful degradation — never crash; fallback returns best-effort
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class CircuitState(str, Enum):
    """The three states of the circuit breaker finite-state machine."""
    CLOSED = "CLOSED"        # Normal operation
    OPEN = "OPEN"            # Failing; reject all calls → fallback
    HALF_OPEN = "HALF_OPEN"  # Probing; one call allowed through


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class CircuitOpenError(RuntimeError):
    """Raised when the circuit breaker is OPEN and no fallback is provided."""

    def __init__(self, breaker_name: str, state: CircuitState) -> None:
        super().__init__(
            f"CircuitBreaker '{breaker_name}' is {state.value}. "
            "No fallback function provided."
        )
        self.breaker_name = breaker_name
        self.state = state


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Async-native circuit breaker with CLOSED / OPEN / HALF_OPEN states.

    Parameters
    ----------
    name:
        Human-readable identifier used in log messages (e.g. ``"researcher"``).
    failure_threshold:
        Number of consecutive failures that trip the breaker from CLOSED to
        OPEN.  Default is 4 (per MF-SAFE-02 specification).
    recovery_timeout:
        Seconds to wait in OPEN state before transitioning to HALF_OPEN to
        attempt a probe call.  Default is 60 (per MF-SAFE-02 specification).
    success_threshold:
        Number of consecutive successes in HALF_OPEN required to return to
        CLOSED.  Default is 1 (single successful probe closes the breaker).
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 4,
        recovery_timeout: float = 60.0,
        success_threshold: int = 1,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._last_failure_time: Optional[float] = None
        self._lock: asyncio.Lock = asyncio.Lock()

        logger.debug(
            "CircuitBreaker '%s' created. failure_threshold=%d "
            "recovery_timeout=%.1fs",
            name, failure_threshold, recovery_timeout
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Current state of the breaker (read-only snapshot)."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Number of consecutive failures since last reset."""
        return self._failure_count

    async def call(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        fallback_func: Optional[Callable[..., Awaitable[Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute *func* with circuit-breaker protection.

        Parameters
        ----------
        func:
            The primary async callable to protect (e.g. an LLM call).
        *args, **kwargs:
            Forwarded to ``func`` and ``fallback_func``.
        fallback_func:
            Async callable invoked when the breaker is OPEN or HALF_OPEN-probe
            fails.  If ``None`` and the breaker is OPEN, raises
            ``CircuitOpenError``.

        Returns
        -------
        Any
            Result of ``func`` on success, or result of ``fallback_func`` when
            the primary is bypassed.
        """
        async with self._lock:
            current_state = await self._resolve_state()

        if current_state is CircuitState.OPEN:
            logger.warning(
                "CircuitBreaker '%s' is OPEN. Routing to fallback.", self.name
            )
            return await self._execute_fallback(fallback_func, *args, **kwargs)

        # CLOSED or HALF_OPEN → attempt the primary call
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as exc:
            await self._on_failure(exc)
            logger.warning(
                "CircuitBreaker '%s' recorded failure #%d/%d: %s",
                self.name, self._failure_count, self.failure_threshold, exc
            )
            if self._state is CircuitState.OPEN:
                # Just tripped — route to fallback if one is provided,
                # otherwise re-raise the original exception (not
                # CircuitOpenError which is only for *already-open* calls).
                if fallback_func is not None:
                    return await self._execute_fallback(
                        fallback_func, *args, **kwargs
                    )
            raise

    def call_sync(
        self,
        func: Callable[..., Any],
        *args: Any,
        fallback_func: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Synchronous convenience wrapper around ``call()``.

        Runs the coroutine in a new event loop.  Intended for scripts and
        tests that do not already have a running loop.  Do NOT use in
        production FastAPI handlers — use ``await call(...)`` instead.
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.call(func, *args, fallback_func=fallback_func, **kwargs)
            )
        finally:
            loop.close()

    def reset(self) -> None:
        """Manually reset the breaker to CLOSED with zero failure count.

        Useful in test teardowns or after a known infrastructure fix.
        """
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        logger.info("CircuitBreaker '%s' manually reset to CLOSED.", self.name)

    def get_status(self) -> dict[str, Any]:
        """Return a snapshot of the breaker's current internal state.

        Returns
        -------
        dict
            Keys: ``name``, ``state``, ``failure_count``, ``success_count``,
            ``last_failure_time``, ``seconds_since_last_failure``.
        """
        now = time.monotonic()
        seconds_open: Optional[float] = (
            now - self._last_failure_time
            if self._last_failure_time is not None
            else None
        )
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_s": self.recovery_timeout,
            "last_failure_time": self._last_failure_time,
            "seconds_since_last_failure": seconds_open,
        }

    # ------------------------------------------------------------------
    # State-machine helpers (called inside _lock)
    # ------------------------------------------------------------------

    async def _resolve_state(self) -> CircuitState:
        """Evaluate whether the breaker should transition OPEN → HALF_OPEN.

        Must be called inside ``self._lock``.
        """
        if self._state is CircuitState.OPEN and self._last_failure_time is not None:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info(
                    "CircuitBreaker '%s' OPEN → HALF_OPEN after %.1fs.",
                    self.name, elapsed
                )
        return self._state

    async def _on_success(self) -> None:
        """Record a successful call and potentially close the breaker."""
        async with self._lock:
            self._success_count += 1
            if self._state is CircuitState.HALF_OPEN:
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._last_failure_time = None
                    logger.info(
                        "CircuitBreaker '%s' HALF_OPEN → CLOSED (probe succeeded).",
                        self.name
                    )
            elif self._state is CircuitState.CLOSED:
                # Successful call — reset failure streak
                self._failure_count = 0

    async def _on_failure(self, exc: Exception) -> None:
        """Record a failed call and potentially open the breaker."""
        async with self._lock:
            self._failure_count += 1
            self._success_count = 0
            self._last_failure_time = time.monotonic()

            if self._state is CircuitState.HALF_OPEN:
                # Probe failed → back to OPEN
                self._state = CircuitState.OPEN
                logger.warning(
                    "CircuitBreaker '%s' HALF_OPEN → OPEN (probe failed: %s).",
                    self.name, exc
                )
            elif (
                self._state is CircuitState.CLOSED
                and self._failure_count >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                logger.error(
                    "CircuitBreaker '%s' CLOSED → OPEN after %d consecutive "
                    "failures. Last error: %s",
                    self.name, self._failure_count, exc
                )

    async def _execute_fallback(
        self,
        fallback_func: Optional[Callable[..., Awaitable[Any]]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run *fallback_func* or raise ``CircuitOpenError`` if none provided."""
        if fallback_func is None:
            raise CircuitOpenError(self.name, self._state)
        try:
            result = await fallback_func(*args, **kwargs)
            logger.info(
                "CircuitBreaker '%s': fallback succeeded.", self.name
            )
            return result
        except Exception as exc:
            logger.error(
                "CircuitBreaker '%s': fallback also failed: %s", self.name, exc
            )
            raise


# ---------------------------------------------------------------------------
# Module-level registry of all active circuit breakers
# ---------------------------------------------------------------------------

# Maps breaker name → CircuitBreaker instance.
# Other modules import this dict to look up role-specific breakers.
_REGISTRY: dict[str, CircuitBreaker] = {}


def get_or_create_breaker(
    name: str,
    failure_threshold: int = 4,
    recovery_timeout: float = 60.0,
    success_threshold: int = 1,
) -> CircuitBreaker:
    """Return an existing breaker by *name* or create and register a new one.

    Parameters
    ----------
    name:
        Unique identifier (e.g. ``"researcher"``, ``"supervisor"``).
    failure_threshold:
        Forwarded to ``CircuitBreaker.__init__``.
    recovery_timeout:
        Forwarded to ``CircuitBreaker.__init__``.
    success_threshold:
        Forwarded to ``CircuitBreaker.__init__``.

    Returns
    -------
    CircuitBreaker
        The shared instance for *name*.
    """
    if name not in _REGISTRY:
        _REGISTRY[name] = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
        )
    return _REGISTRY[name]


def all_breaker_statuses() -> list[dict[str, Any]]:
    """Return status snapshots for all registered circuit breakers."""
    return [b.get_status() for b in _REGISTRY.values()]
