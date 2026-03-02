"""
masis.config.settings
=====================
Environment variable loading, validation, and the Settings singleton.

Implements
----------
ENG-02 / M3 / S1b : Load dotenv, validate all required keys are present.
MF-API-08         : Centralised config with env var overrides.

Behaviour
---------
1. At import time, python-dotenv loads variables from .env (if present).
   This is safe to call multiple times -- subsequent calls are no-ops.
2. Settings() reads all env vars and validates required ones are non-empty.
3. Missing required env vars raise EnvironmentError with a clear list of
   every missing variable -- never a cryptic KeyError at runtime.
4. get_settings() returns the process-level singleton (lazy init on first call).

Usage
-----
    # In application startup (e.g. FastAPI lifespan):
    from masis.config.settings import get_settings, validate_env

    validate_env()          # Raises EnvironmentError if any required key is missing
    settings = get_settings()
    print(settings.openai_api_key[:8])  # "sk-proj-"

    # In agent code:
    from masis.config.settings import get_settings
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------
# We load python-dotenv here so any module importing masis.config.settings
# automatically gets env vars from .env populated before Settings() is built.


def _load_dotenv() -> bool:
    """
    Attempt to load .env using python-dotenv.

    Searches for .env in the following order:
    1. Current working directory
    2. Parent directory of this file (masis/)
    3. Two parents up (project root, where .env.example lives)

    Returns:
        True if a .env file was found and loaded, False otherwise.
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import]
    except ImportError:
        logger.debug(
            "python-dotenv not installed. Env vars must be set in the shell environment."
        )
        return False

    candidates = [
        Path.cwd() / ".env",
        Path(__file__).parent / ".env",      # masis/config/.env (uncommon)
        Path(__file__).parent.parent / ".env",  # masis/.env
        Path(__file__).parent.parent.parent / ".env",  # project root .env
    ]

    for candidate in candidates:
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)
            logger.debug("Loaded environment variables from %s", candidate)
            return True

    logger.debug("No .env file found in any standard location.")
    return False


# Load on module import
_DOTENV_LOADED = _load_dotenv()

# ---------------------------------------------------------------------------
# Required and optional environment variables
# ---------------------------------------------------------------------------

_REQUIRED_ENV_VARS: List[str] = [
    "OPENAI_API_KEY",
    # POSTGRES_URL is required in production but optional in dev (InMemorySaver)
    # TAVILY_API_KEY is required only if web_search tasks are enabled
]

_OPTIONAL_ENV_VARS: List[str] = [
    "POSTGRES_URL",
    "TAVILY_API_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_PROJECT",
    "MODEL_SUPERVISOR",
    "MODEL_RESEARCHER",
    "MODEL_SKEPTIC",
    "MODEL_SYNTHESIZER",
    "MODEL_AMBIGUITY",
    "MODEL_EMBEDDER",
    "RESEARCHER_MAX_PARALLEL",
    "RESEARCHER_MAX_TOTAL",
    "RESEARCHER_TIMEOUT_S",
    "WEB_SEARCH_MAX_PARALLEL",
    "WEB_SEARCH_MAX_TOTAL",
    "WEB_SEARCH_TIMEOUT_S",
    "SKEPTIC_MAX_PARALLEL",
    "SKEPTIC_MAX_TOTAL",
    "SKEPTIC_TIMEOUT_S",
    "SYNTHESIZER_MAX_PARALLEL",
    "SYNTHESIZER_MAX_TOTAL",
    "SYNTHESIZER_TIMEOUT_S",
    "CHROMA_PERSIST_DIRECTORY",
    "LOG_LEVEL",
    "ENVIRONMENT",  # "development" | "staging" | "production"
]


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------

class Settings:
    """
    Process-level configuration loaded from environment variables.

    All attributes are read once at construction time. The singleton
    pattern (get_settings()) avoids re-reading env vars on every call.

    Attributes
    ----------
    openai_api_key     : OpenAI API key (required)
    postgres_url       : PostgreSQL connection string for checkpointing (optional in dev)
    tavily_api_key     : Tavily web search API key (optional -- required for web_search tasks)
    langfuse_secret_key: Langfuse tracing secret key (optional -- observability)
    langfuse_public_key: Langfuse tracing public key (optional)
    langsmith_api_key  : LangSmith API key (optional -- alternative observability)
    langchain_tracing_v2: Enable LangChain tracing ("true"/"false")
    langchain_project  : LangChain project name for tracing
    model_supervisor   : Model override for Supervisor (default: gpt-4.1)
    model_researcher   : Model override for Researcher (default: gpt-4.1-mini)
    model_skeptic      : Model override for Skeptic LLM judge (default: o3-mini)
    model_synthesizer  : Model override for Synthesizer (default: gpt-4.1)
    model_ambiguity    : Model override for Ambiguity Detector (default: gpt-4.1-mini)
    model_embedder     : Embedding model (default: text-embedding-3-small)
    chroma_persist_dir : ChromaDB persistence directory (default: ./chroma_db)
    log_level          : Logging level string (default: INFO)
    environment        : Deployment environment (default: development)
    is_production      : True when environment == "production"
    is_development     : True when environment == "development" or "" (default)
    """

    def __init__(self) -> None:
        # Required
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

        # Optional but important
        self.postgres_url: str = os.getenv("POSTGRES_URL", "")
        self.tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

        # Observability
        self.langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
        self.langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
        self.langchain_tracing_v2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
        self.langchain_project: str = os.getenv("LANGCHAIN_PROJECT", "masis")

        # Model routing overrides
        self.model_supervisor: str = os.getenv("MODEL_SUPERVISOR", "gpt-4.1")
        self.model_researcher: str = os.getenv("MODEL_RESEARCHER", "gpt-4.1-mini")
        self.model_skeptic: str = os.getenv("MODEL_SKEPTIC", "o3-mini")
        self.model_synthesizer: str = os.getenv("MODEL_SYNTHESIZER", "gpt-4.1")
        self.model_ambiguity: str = os.getenv("MODEL_AMBIGUITY", "gpt-4.1-mini")
        self.model_embedder: str = os.getenv("MODEL_EMBEDDER", "text-embedding-3-small")

        # Infrastructure
        self.chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

        # Runtime configuration
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
        self.environment: str = os.getenv("ENVIRONMENT", "development").lower()

    @property
    def is_production(self) -> bool:
        """True when running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """True when running in development or test environment."""
        return self.environment in ("development", "dev", "test", "")

    @property
    def use_postgres_checkpointer(self) -> bool:
        """True when a PostgreSQL URL is configured (enables PostgresSaver)."""
        return bool(self.postgres_url)

    @property
    def use_langfuse(self) -> bool:
        """True when Langfuse credentials are configured."""
        return bool(self.langfuse_secret_key and self.langfuse_public_key)

    @property
    def use_langsmith(self) -> bool:
        """True when LangSmith tracing is configured."""
        return bool(self.langsmith_api_key) or self.langchain_tracing_v2 == "true"

    def validate(self) -> None:
        """
        Validate that all required environment variables are set.

        Raises:
            EnvironmentError: If any required variable is missing or empty.
                              The error message lists ALL missing variables at once
                              so the user can fix them in a single iteration.

        Example:
            >>> settings = Settings()
            >>> settings.validate()
            EnvironmentError: Missing required environment variables:
              - OPENAI_API_KEY: Required for all LLM calls.
            Set these variables in your .env file or shell environment.
        """
        missing: List[str] = []

        if not self.openai_api_key:
            missing.append(
                "OPENAI_API_KEY: Required for all LLM calls "
                "(Supervisor, Researcher, Skeptic, Synthesizer)."
            )

        if self.is_production and not self.postgres_url:
            missing.append(
                "POSTGRES_URL: Required in production for PostgresSaver checkpointing. "
                "Format: postgresql://user:password@host:5432/dbname"
            )

        if missing:
            lines = "\n  - ".join(missing)
            raise EnvironmentError(
                f"Missing required environment variables:\n  - {lines}\n"
                "Set these variables in your .env file or shell environment.\n"
                f"See masis/.env.example for the full list of supported variables."
            )

    def summary(self) -> str:
        """
        Return a human-readable summary of the active configuration.

        Sensitive values are masked. Safe to log at startup.
        """
        def mask(v: str) -> str:
            if not v:
                return "(not set)"
            if len(v) <= 8:
                return "****"
            return v[:4] + "****" + v[-4:]

        lines = [
            "MASIS Configuration Summary",
            "=" * 40,
            f"  environment:          {self.environment}",
            f"  openai_api_key:       {mask(self.openai_api_key)}",
            f"  postgres_url:         {mask(self.postgres_url)}",
            f"  tavily_api_key:       {mask(self.tavily_api_key)}",
            f"  model_supervisor:     {self.model_supervisor}",
            f"  model_researcher:     {self.model_researcher}",
            f"  model_skeptic:        {self.model_skeptic}",
            f"  model_synthesizer:    {self.model_synthesizer}",
            f"  model_ambiguity:      {self.model_ambiguity}",
            f"  model_embedder:       {self.model_embedder}",
            f"  chroma_persist_dir:   {self.chroma_persist_dir}",
            f"  log_level:            {self.log_level}",
            f"  use_postgres:         {self.use_postgres_checkpointer}",
            f"  use_langfuse:         {self.use_langfuse}",
            f"  use_langsmith:        {self.use_langsmith}",
            "=" * 40,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_settings_singleton: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Return the process-level Settings singleton.

    The singleton is created on the first call and reused thereafter.
    This means env vars are read once at startup, which is the correct behaviour
    for long-running services (changes to .env require a restart).

    Returns:
        The singleton Settings instance.

    Example:
        >>> settings = get_settings()
        >>> settings.openai_api_key
        'sk-proj-...'
    """
    global _settings_singleton
    if _settings_singleton is None:
        _settings_singleton = Settings()
    return _settings_singleton


def validate_env() -> None:
    """
    Validate the environment and raise EnvironmentError if required vars are missing.

    This is the recommended call at application startup:
        from masis.config.settings import validate_env
        validate_env()  # Fail fast with a clear error before any LLM calls

    Raises:
        EnvironmentError: Lists ALL missing required variables.
    """
    get_settings().validate()


def reset_settings() -> None:
    """
    Reset the settings singleton. Used in tests to reload env vars between cases.

    This is ONLY for testing -- do not call in production code.

    Example:
        >>> import os
        >>> os.environ["OPENAI_API_KEY"] = "sk-test"
        >>> reset_settings()
        >>> get_settings().openai_api_key
        'sk-test'
    """
    global _settings_singleton
    _settings_singleton = None
    _load_dotenv()
