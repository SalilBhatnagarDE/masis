"""
tests/test_circuit_breaker.py
==============================
Comprehensive unit and integration tests for masis.infra.circuit_breaker.

Covers
------
- State transitions: CLOSED → OPEN, OPEN → HALF_OPEN, HALF_OPEN → CLOSED,
  HALF_OPEN → OPEN (probe failure).
- Fallback invocation when OPEN.
- CircuitOpenError when OPEN and no fallback provided.
- get_or_create_breaker registry.
- call_sync convenience wrapper.
- get_status() snapshot structure.
- all_breaker_statuses() list.
- Concurrency: multiple concurrent calls do not corrupt state.
"""

from __future__ import annotations

import asyncio
import time
import pytest

from masis.infra.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    all_breaker_statuses,
    get_or_create_breaker,
    _REGISTRY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _ok(*args, **kwargs) -> str:
    """Always succeeds."""
    return "ok"


async def _fail(*args, **kwargs) -> None:
    """Always raises."""
    raise ValueError("deliberate failure")


async def _fallback(*args, **kwargs) -> str:
    """Fallback that always succeeds."""
    return "fallback_result"


# ---------------------------------------------------------------------------
# Basic state-machine tests
# ---------------------------------------------------------------------------

class TestCircuitBreakerStateMachine:

    def setup_method(self):
        """Fresh breaker for each test."""
        self.cb = CircuitBreaker(
            name="test_breaker",
            failure_threshold=4,
            recovery_timeout=60.0,
        )

    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self):
        assert self.cb.state is CircuitState.CLOSED
        assert self.cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_success_keeps_closed(self):
        result = await self.cb.call(_ok)
        assert result == "ok"
        assert self.cb.state is CircuitState.CLOSED
        assert self.cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_single_failure_does_not_open(self):
        with pytest.raises(ValueError):
            await self.cb.call(_fail)
        assert self.cb.state is CircuitState.CLOSED
        assert self.cb.failure_count == 1

    @pytest.mark.asyncio
    async def test_four_failures_open_breaker(self):
        """MF-SAFE-02: 4 consecutive failures → OPEN."""
        for _ in range(3):
            with pytest.raises(ValueError):
                await self.cb.call(_fail)
        assert self.cb.state is CircuitState.CLOSED

        # 4th failure should trip the breaker
        result = await self.cb.call(_fail, fallback_func=_fallback)
        assert result == "fallback_result"
        assert self.cb.state is CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_breaker_uses_fallback_immediately(self):
        """Once OPEN, all calls should go to fallback without calling primary."""
        call_count = {"primary": 0}

        async def primary():
            call_count["primary"] += 1
            raise ValueError("failure")

        # Trip the breaker
        for _ in range(4):
            with pytest.raises((ValueError, Exception)):
                await self.cb.call(primary)

        self.cb._state = CircuitState.OPEN  # Force OPEN for clarity

        # Next call should hit fallback without calling primary
        call_count_before = call_count["primary"]
        result = await self.cb.call(primary, fallback_func=_fallback)
        assert result == "fallback_result"
        assert call_count["primary"] == call_count_before  # primary not called

    @pytest.mark.asyncio
    async def test_open_raises_without_fallback(self):
        """OPEN breaker with no fallback → CircuitOpenError."""
        self.cb._state = CircuitState.OPEN
        self.cb._last_failure_time = time.monotonic()
        with pytest.raises(CircuitOpenError) as exc_info:
            await self.cb.call(_ok)
        assert "test_breaker" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_open_transitions_to_half_open_after_timeout(self):
        """MF-SAFE-02: After recovery_timeout → HALF_OPEN."""
        cb = CircuitBreaker(
            name="timeout_test",
            failure_threshold=2,
            recovery_timeout=0.01,  # 10ms for fast test
        )
        # Trip the breaker
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(_fail)
        assert cb.state is CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.05)

        # Next call should probe (HALF_OPEN)
        result = await cb.call(_ok)
        assert result == "ok"
        assert cb.state is CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_probe_failure_returns_to_open(self):
        """HALF_OPEN probe failure → back to OPEN."""
        cb = CircuitBreaker(
            name="probe_fail_test",
            failure_threshold=2,
            recovery_timeout=0.01,
        )
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(_fail)
        await asyncio.sleep(0.05)

        # Probe fails → back to OPEN
        with pytest.raises(Exception):
            await cb.call(_fail)
        assert cb.state is CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self):
        """Successful call after partial failures resets the failure streak."""
        cb = CircuitBreaker(name="reset_test", failure_threshold=4)
        for _ in range(3):
            with pytest.raises(ValueError):
                await cb.call(_fail)
        assert cb.failure_count == 3

        await cb.call(_ok)  # Success
        assert cb.failure_count == 0
        assert cb.state is CircuitState.CLOSED


# ---------------------------------------------------------------------------
# Fallback chain tests
# ---------------------------------------------------------------------------

class TestFallbackBehaviour:

    @pytest.mark.asyncio
    async def test_fallback_called_when_open(self):
        cb = CircuitBreaker(name="fb_test", failure_threshold=1)
        with pytest.raises(ValueError):
            await cb.call(_fail)
        assert cb.state is CircuitState.OPEN

        result = await cb.call(_fail, fallback_func=_fallback)
        assert result == "fallback_result"

    @pytest.mark.asyncio
    async def test_fallback_failure_propagates(self):
        """If both primary and fallback fail, exception from fallback propagates."""
        cb = CircuitBreaker(name="both_fail_test", failure_threshold=1)
        with pytest.raises(ValueError):
            await cb.call(_fail)

        async def bad_fallback():
            raise RuntimeError("fallback also failed")

        with pytest.raises(RuntimeError, match="fallback also failed"):
            await cb.call(_fail, fallback_func=bad_fallback)


# ---------------------------------------------------------------------------
# Reset and status tests
# ---------------------------------------------------------------------------

class TestResetAndStatus:

    def test_reset_returns_to_closed(self):
        cb = CircuitBreaker(name="reset_cb", failure_threshold=4)
        cb._state = CircuitState.OPEN
        cb._failure_count = 4
        cb._last_failure_time = time.monotonic()

        cb.reset()

        assert cb.state is CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb._last_failure_time is None

    def test_get_status_fields(self):
        cb = CircuitBreaker(name="status_cb", failure_threshold=4, recovery_timeout=60.0)
        status = cb.get_status()

        assert status["name"] == "status_cb"
        assert status["state"] == "CLOSED"
        assert status["failure_count"] == 0
        assert status["failure_threshold"] == 4
        assert status["recovery_timeout_s"] == 60.0
        assert "seconds_since_last_failure" in status


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:

    def setup_method(self):
        # Clear relevant entries in registry
        _REGISTRY.pop("registry_test_a", None)
        _REGISTRY.pop("registry_test_b", None)

    def test_get_or_create_creates_new(self):
        cb = get_or_create_breaker("registry_test_a", failure_threshold=5)
        assert cb.name == "registry_test_a"
        assert cb.failure_threshold == 5

    def test_get_or_create_returns_same_instance(self):
        cb1 = get_or_create_breaker("registry_test_b")
        cb2 = get_or_create_breaker("registry_test_b")
        assert cb1 is cb2

    def test_all_breaker_statuses_returns_list(self):
        get_or_create_breaker("registry_test_a")
        statuses = all_breaker_statuses()
        assert isinstance(statuses, list)
        assert any(s["name"] == "registry_test_a" for s in statuses)


# ---------------------------------------------------------------------------
# call_sync wrapper
# ---------------------------------------------------------------------------

class TestCallSync:

    def test_call_sync_success(self):
        cb = CircuitBreaker(name="sync_test", failure_threshold=4)
        result = cb.call_sync(_ok)
        assert result == "ok"

    def test_call_sync_failure_propagates(self):
        cb = CircuitBreaker(name="sync_fail_test", failure_threshold=4)
        with pytest.raises(ValueError):
            cb.call_sync(_fail)


# ---------------------------------------------------------------------------
# Concurrency test
# ---------------------------------------------------------------------------

class TestConcurrency:

    @pytest.mark.asyncio
    async def test_concurrent_calls_do_not_corrupt_state(self):
        """20 concurrent calls — all succeed — breaker stays CLOSED."""
        cb = CircuitBreaker(name="concurrent_test", failure_threshold=4)

        async def fast_ok():
            await asyncio.sleep(0.001)
            return "ok"

        tasks = [cb.call(fast_ok) for _ in range(20)]
        results = await asyncio.gather(*tasks)
        assert all(r == "ok" for r in results)
        assert cb.state is CircuitState.CLOSED
        assert cb.failure_count == 0


# ---------------------------------------------------------------------------
# CircuitOpenError
# ---------------------------------------------------------------------------

class TestCircuitOpenError:

    def test_error_message_contains_name(self):
        err = CircuitOpenError("my_breaker", CircuitState.OPEN)
        assert "my_breaker" in str(err)
        assert "OPEN" in str(err)

    def test_error_attributes(self):
        err = CircuitOpenError("role_x", CircuitState.HALF_OPEN)
        assert err.breaker_name == "role_x"
        assert err.state is CircuitState.HALF_OPEN
