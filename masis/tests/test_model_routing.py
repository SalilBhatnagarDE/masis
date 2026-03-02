"""
test_model_routing.py
=====================
Unit tests for masis.config.model_routing

Covers ENG-02 / M1 / S1 and S2:
  - MODEL_ROUTING dict — defaults and env-var override
  - FALLBACK_CHAINS dict — all roles present, chain structure correct
  - TOOL_LIMITS dict — all agent types covered with correct values
  - get_model() — default, override, unknown role
  - get_fallback() — next in chain, end of chain returns None, unknown role
  - estimate_cost() — correct USD calculation
  - get_rate_limit() — accessor for TOOL_LIMITS values

Run:
    pytest masis/tests/test_model_routing.py -v
"""

from __future__ import annotations

import os
import importlib
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reload_routing():
    """Re-import model_routing after env-var changes so the module re-reads os.getenv."""
    import masis.config.model_routing as m
    importlib.reload(m)
    return m


# ---------------------------------------------------------------------------
# MODEL_ROUTING defaults
# ---------------------------------------------------------------------------

class TestModelRoutingDefaults:
    """Test that MODEL_ROUTING contains expected default values when no env vars set."""

    def setup_method(self):
        # Remove overrides so defaults are used
        for key in ("MODEL_SUPERVISOR", "MODEL_RESEARCHER", "MODEL_SKEPTIC",
                    "MODEL_SYNTHESIZER", "MODEL_AMBIGUITY", "MODEL_EMBEDDER"):
            os.environ.pop(key, None)

    def test_all_required_roles_present(self):
        from masis.config.model_routing import MODEL_ROUTING
        required = {
            "supervisor_plan", "supervisor_slow", "researcher",
            "skeptic_llm", "synthesizer", "ambiguity_detector", "embedder",
        }
        for role in required:
            assert role in MODEL_ROUTING, f"MODEL_ROUTING missing role '{role}'"

    def test_supervisor_default(self):
        # Fresh import with no env vars → should be gpt-4.1
        mod = reload_routing()
        assert mod.MODEL_ROUTING["supervisor_plan"] == "gpt-4.1"
        assert mod.MODEL_ROUTING["supervisor_slow"] == "gpt-4.1"

    def test_researcher_default(self):
        mod = reload_routing()
        assert mod.MODEL_ROUTING["researcher"] == "gpt-4.1-mini"

    def test_skeptic_default(self):
        mod = reload_routing()
        assert mod.MODEL_ROUTING["skeptic_llm"] == "o3-mini"

    def test_synthesizer_default(self):
        mod = reload_routing()
        assert mod.MODEL_ROUTING["synthesizer"] == "gpt-4.1"

    def test_ambiguity_detector_default(self):
        mod = reload_routing()
        assert mod.MODEL_ROUTING["ambiguity_detector"] == "gpt-4.1-mini"

    def test_embedder_default(self):
        mod = reload_routing()
        assert mod.MODEL_ROUTING["embedder"] == "text-embedding-3-small"


# ---------------------------------------------------------------------------
# MODEL_ROUTING env-var overrides
# ---------------------------------------------------------------------------

class TestModelRoutingEnvOverrides:
    """Test that MODEL_ROUTING picks up environment variable overrides."""

    def teardown_method(self):
        # Clean up env vars set during tests
        for key in ("MODEL_SUPERVISOR", "MODEL_RESEARCHER", "MODEL_SKEPTIC",
                    "MODEL_SYNTHESIZER", "MODEL_AMBIGUITY", "MODEL_EMBEDDER"):
            os.environ.pop(key, None)

    def test_researcher_env_override(self):
        os.environ["MODEL_RESEARCHER"] = "gpt-4.1-nano"
        mod = reload_routing()
        assert mod.MODEL_ROUTING["researcher"] == "gpt-4.1-nano"

    def test_supervisor_env_override(self):
        os.environ["MODEL_SUPERVISOR"] = "gpt-4.1-mini"
        mod = reload_routing()
        assert mod.MODEL_ROUTING["supervisor_plan"] == "gpt-4.1-mini"
        assert mod.MODEL_ROUTING["supervisor_slow"] == "gpt-4.1-mini"

    def test_skeptic_env_override(self):
        os.environ["MODEL_SKEPTIC"] = "gpt-4.1"
        mod = reload_routing()
        assert mod.MODEL_ROUTING["skeptic_llm"] == "gpt-4.1"


# ---------------------------------------------------------------------------
# get_model()
# ---------------------------------------------------------------------------

class TestGetModel:
    """Test the get_model() accessor function."""

    def setup_method(self):
        for key in ("MODEL_RESEARCHER", "MODEL_SUPERVISOR"):
            os.environ.pop(key, None)
        reload_routing()

    def teardown_method(self):
        for key in ("MODEL_RESEARCHER", "MODEL_SUPERVISOR"):
            os.environ.pop(key, None)

    def test_default_researcher(self):
        from masis.config.model_routing import get_model
        assert get_model("researcher") == "gpt-4.1-mini"

    def test_override_parameter(self):
        """ENG-02 M1 S1b: get_model("researcher", "gpt-4.1-nano") → "gpt-4.1-nano"."""
        from masis.config.model_routing import get_model
        result = get_model("researcher", "gpt-4.1-nano")
        assert result == "gpt-4.1-nano"

    def test_override_takes_priority_over_env(self):
        os.environ["MODEL_RESEARCHER"] = "gpt-4.1"
        reload_routing()
        from masis.config.model_routing import get_model
        # Explicit override wins over env var
        assert get_model("researcher", "gpt-4.1-nano") == "gpt-4.1-nano"

    def test_unknown_role_returns_empty_string(self):
        from masis.config.model_routing import get_model
        result = get_model("nonexistent_role")
        assert result == ""

    def test_env_var_override(self):
        """ENG-02 M1 S1b test: Set env var → verify get_model picks it up after reload."""
        os.environ["MODEL_RESEARCHER"] = "test-model-xyz"
        mod = reload_routing()
        assert mod.get_model("researcher") == "test-model-xyz"

    def test_none_override_uses_routing(self):
        from masis.config.model_routing import get_model
        result = get_model("synthesizer", None)
        assert result == "gpt-4.1"


# ---------------------------------------------------------------------------
# FALLBACK_CHAINS
# ---------------------------------------------------------------------------

class TestFallbackChains:
    """Test that FALLBACK_CHAINS are properly structured per MF-SAFE-03."""

    def test_all_roles_have_chains(self):
        from masis.config.model_routing import FALLBACK_CHAINS
        required_roles = {"researcher", "supervisor", "skeptic_llm", "synthesizer"}
        for role in required_roles:
            assert role in FALLBACK_CHAINS, f"FALLBACK_CHAINS missing role '{role}'"

    def test_chains_are_lists(self):
        from masis.config.model_routing import FALLBACK_CHAINS
        for role, chain in FALLBACK_CHAINS.items():
            assert isinstance(chain, list), f"Chain for '{role}' must be a list"

    def test_chains_have_at_least_one_entry(self):
        from masis.config.model_routing import FALLBACK_CHAINS
        for role, chain in FALLBACK_CHAINS.items():
            assert len(chain) >= 1, f"Chain for '{role}' must have at least 1 entry"

    def test_researcher_chain_has_fallback(self):
        from masis.config.model_routing import FALLBACK_CHAINS
        chain = FALLBACK_CHAINS["researcher"]
        assert len(chain) >= 2, "Researcher must have at least primary + 1 fallback"
        assert chain[1] == "gpt-4.1"

    def test_skeptic_chain_has_fallback(self):
        from masis.config.model_routing import FALLBACK_CHAINS
        chain = FALLBACK_CHAINS["skeptic_llm"]
        assert len(chain) >= 2
        assert chain[1] == "gpt-4.1"


# ---------------------------------------------------------------------------
# get_fallback()
# ---------------------------------------------------------------------------

class TestGetFallback:
    """Test get_fallback() — next model in chain after primary fails."""

    def setup_method(self):
        # Reset to defaults
        for key in ("MODEL_RESEARCHER", "MODEL_SKEPTIC"):
            os.environ.pop(key, None)
        reload_routing()

    def teardown_method(self):
        for key in ("MODEL_RESEARCHER", "MODEL_SKEPTIC"):
            os.environ.pop(key, None)

    def test_researcher_first_fallback(self):
        """ENG-02 M1 S2b: get_fallback("researcher") returns "gpt-4.1"."""
        from masis.config.model_routing import get_fallback
        result = get_fallback("researcher")
        assert result == "gpt-4.1"

    def test_fallback_after_current_model(self):
        """get_fallback("researcher", "gpt-4.1-mini") should return "gpt-4.1"."""
        from masis.config.model_routing import get_fallback
        result = get_fallback("researcher", "gpt-4.1-mini")
        assert result == "gpt-4.1"

    def test_end_of_chain_returns_none(self):
        """After the last model in the chain, get_fallback returns None."""
        from masis.config.model_routing import FALLBACK_CHAINS, get_fallback
        chain = FALLBACK_CHAINS["researcher"]
        last_model = chain[-1]
        result = get_fallback("researcher", last_model)
        assert result is None

    def test_unknown_role_returns_none(self):
        from masis.config.model_routing import get_fallback
        result = get_fallback("nonexistent_role")
        assert result is None

    def test_skeptic_fallback(self):
        from masis.config.model_routing import get_fallback
        result = get_fallback("skeptic_llm")
        assert result == "gpt-4.1"

    def test_synthesizer_fallback(self):
        from masis.config.model_routing import get_fallback
        result = get_fallback("synthesizer")
        assert result is not None  # synthesizer must have at least one fallback


# ---------------------------------------------------------------------------
# TOOL_LIMITS
# ---------------------------------------------------------------------------

class TestToolLimitsRouting:
    """Test that TOOL_LIMITS in model_routing match the spec from ENG-02 M2 S1."""

    def test_all_agent_types_present(self):
        from masis.config.model_routing import TOOL_LIMITS
        for agent in ("researcher", "web_search", "skeptic", "synthesizer"):
            assert agent in TOOL_LIMITS, f"TOOL_LIMITS missing '{agent}'"

    def test_researcher_limits_match_spec(self):
        from masis.config.model_routing import TOOL_LIMITS
        r = TOOL_LIMITS["researcher"]
        assert r["max_parallel"] == 3
        assert r["max_total"] == 8
        assert r["timeout_s"] == 30

    def test_web_search_limits(self):
        from masis.config.model_routing import TOOL_LIMITS
        w = TOOL_LIMITS["web_search"]
        assert w["max_parallel"] == 2
        assert w["max_total"] == 4
        assert w["timeout_s"] == 15

    def test_skeptic_limits(self):
        from masis.config.model_routing import TOOL_LIMITS
        s = TOOL_LIMITS["skeptic"]
        assert s["max_parallel"] == 1
        assert s["max_total"] == 3
        assert s["timeout_s"] == 45

    def test_synthesizer_limits(self):
        from masis.config.model_routing import TOOL_LIMITS
        s = TOOL_LIMITS["synthesizer"]
        assert s["max_parallel"] == 1
        assert s["max_total"] == 3
        assert s["timeout_s"] == 60


# ---------------------------------------------------------------------------
# estimate_cost()
# ---------------------------------------------------------------------------

class TestEstimateCost:
    """Test the cost estimation utility."""

    def test_gpt4o_cost_calculation(self):
        from masis.config.model_routing import estimate_cost
        # 1000 input * $0.005/1K + 500 output * $0.015/1K = $0.005 + $0.0075 = $0.0125
        cost = estimate_cost("gpt-4.1", 1000, 500)
        assert abs(cost - 0.0125) < 1e-6

    def test_unknown_model_returns_zero(self):
        from masis.config.model_routing import estimate_cost
        cost = estimate_cost("nonexistent-model", 10000, 5000)
        assert cost == 0.0

    def test_zero_tokens_returns_zero(self):
        from masis.config.model_routing import estimate_cost
        cost = estimate_cost("gpt-4.1", 0, 0)
        assert cost == 0.0

    def test_gpt4o_mini_cheaper_than_gpt4o(self):
        from masis.config.model_routing import estimate_cost
        cost_mini = estimate_cost("gpt-4.1-mini", 1000, 500)
        cost_full = estimate_cost("gpt-4.1", 1000, 500)
        assert cost_mini < cost_full

    def test_embedding_model_output_cost_zero(self):
        from masis.config.model_routing import estimate_cost
        # Embedding models have no output cost
        cost_input_only = estimate_cost("text-embedding-3-small", 1000, 0)
        cost_with_output = estimate_cost("text-embedding-3-small", 1000, 1000)
        assert cost_input_only == cost_with_output  # output tokens don't add cost


# ---------------------------------------------------------------------------
# get_rate_limit()
# ---------------------------------------------------------------------------

class TestGetRateLimit:
    """Test the get_rate_limit() accessor."""

    def test_researcher_max_parallel(self):
        from masis.config.model_routing import get_rate_limit
        assert get_rate_limit("researcher", "max_parallel") == 3

    def test_skeptic_timeout(self):
        from masis.config.model_routing import get_rate_limit
        assert get_rate_limit("skeptic", "timeout_s") == 45

    def test_unknown_agent_returns_none(self):
        from masis.config.model_routing import get_rate_limit
        assert get_rate_limit("unknown_agent", "max_parallel") is None

    def test_unknown_key_returns_none(self):
        from masis.config.model_routing import get_rate_limit
        assert get_rate_limit("researcher", "nonexistent_key") is None
