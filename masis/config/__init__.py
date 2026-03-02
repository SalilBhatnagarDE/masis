"""
masis.config
============
Configuration layer for the MASIS system.

This package centralises all environment-dependent configuration:
model selection, fallback chains, per-agent rate limits, and environment
variable loading with validation.

Exports -- model_routing
------------------------
MODEL_ROUTING    -- dict mapping role -> model identifier (env-var overridable)
FALLBACK_CHAINS  -- dict mapping role -> ordered list of fallback models (MF-SAFE-03)
TOOL_LIMITS      -- dict mapping agent type -> rate limit config (MF-SAFE-05)
get_model()      -- get the configured model for a role, with optional override
get_fallback()   -- get the next fallback model for a role (MF-SAFE-03)

Exports -- settings
-------------------
Settings          -- Pydantic settings model that loads and validates all env vars
get_settings()    -- returns the singleton Settings instance
validate_env()    -- raises EnvironmentError listing all missing required variables
"""

from masis.config.model_routing import (
    FALLBACK_CHAINS,
    MODEL_ROUTING,
    TOOL_LIMITS,
    get_fallback,
    get_model,
)
from masis.config.settings import Settings, get_settings, validate_env

__all__ = [
    "FALLBACK_CHAINS",
    "MODEL_ROUTING",
    "TOOL_LIMITS",
    "get_fallback",
    "get_model",
    "Settings",
    "get_settings",
    "validate_env",
]
