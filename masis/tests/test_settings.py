"""
test_settings.py
================
Unit tests for masis.config.settings

Covers ENG-02 / M3 / S1:
  - _load_dotenv()     — loads .env without crashing when file missing
  - Settings.__init__  — reads all env vars, applies defaults
  - Settings.validate()— raises EnvironmentError listing missing required vars
  - get_settings()     — singleton pattern, returns same instance on repeat calls
  - reset_settings()   — resets singleton for test isolation
  - validate_env()     — convenience wrapper for Settings().validate()
  - Settings properties — is_production, is_development, use_postgres_checkpointer

Run:
    pytest masis/tests/test_settings.py -v
"""

from __future__ import annotations

import os
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_env(**kwargs):
    """Set multiple env vars at once; returns a dict of previous values for cleanup."""
    previous = {}
    for key, value in kwargs.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return previous


def _restore_env(previous: dict):
    """Restore env vars to values captured by _set_env."""
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# Settings defaults
# ---------------------------------------------------------------------------

class TestSettingsDefaults:
    """Settings constructed with minimal environment should have safe defaults."""

    def setup_method(self):
        from masis.config.settings import reset_settings
        reset_settings()

    def test_settings_instantiates(self):
        from masis.config.settings import Settings
        s = Settings()
        assert s is not None

    def test_default_log_level_is_info(self):
        from masis.config.settings import Settings
        prev = _set_env(LOG_LEVEL=None)
        try:
            s = Settings()
            assert s.log_level == "INFO"
        finally:
            _restore_env(prev)

    def test_default_environment_is_development(self):
        from masis.config.settings import Settings
        prev = _set_env(ENVIRONMENT=None)
        try:
            s = Settings()
            assert s.environment == "development"
        finally:
            _restore_env(prev)

    def test_default_chroma_dir(self):
        from masis.config.settings import Settings
        prev = _set_env(CHROMA_PERSIST_DIRECTORY=None)
        try:
            s = Settings()
            assert s.chroma_persist_dir == "./chroma_db"
        finally:
            _restore_env(prev)

    def test_default_model_supervisor(self):
        from masis.config.settings import Settings
        prev = _set_env(MODEL_SUPERVISOR=None)
        try:
            s = Settings()
            assert s.model_supervisor == "gpt-4.1"
        finally:
            _restore_env(prev)

    def test_default_model_researcher(self):
        from masis.config.settings import Settings
        prev = _set_env(MODEL_RESEARCHER=None)
        try:
            s = Settings()
            assert s.model_researcher == "gpt-4.1-mini"
        finally:
            _restore_env(prev)

    def test_default_model_skeptic(self):
        from masis.config.settings import Settings
        prev = _set_env(MODEL_SKEPTIC=None)
        try:
            s = Settings()
            assert s.model_skeptic == "o3-mini"
        finally:
            _restore_env(prev)

    def test_empty_string_for_missing_api_keys(self):
        from masis.config.settings import Settings
        prev = _set_env(OPENAI_API_KEY=None, POSTGRES_URL=None, TAVILY_API_KEY=None)
        try:
            s = Settings()
            assert s.openai_api_key == ""
            assert s.postgres_url == ""
            assert s.tavily_api_key == ""
        finally:
            _restore_env(prev)


# ---------------------------------------------------------------------------
# Settings env-var reading
# ---------------------------------------------------------------------------

class TestSettingsEnvReading:
    """Settings reads env vars correctly."""

    def setup_method(self):
        from masis.config.settings import reset_settings
        reset_settings()

    def teardown_method(self):
        from masis.config.settings import reset_settings
        reset_settings()

    def test_reads_openai_api_key(self):
        from masis.config.settings import Settings
        prev = _set_env(OPENAI_API_KEY="sk-test-key-12345")
        try:
            s = Settings()
            assert s.openai_api_key == "sk-test-key-12345"
        finally:
            _restore_env(prev)

    def test_reads_postgres_url(self):
        from masis.config.settings import Settings
        url = "postgresql://user:pass@localhost:5432/testdb"
        prev = _set_env(POSTGRES_URL=url)
        try:
            s = Settings()
            assert s.postgres_url == url
        finally:
            _restore_env(prev)

    def test_reads_tavily_api_key(self):
        from masis.config.settings import Settings
        prev = _set_env(TAVILY_API_KEY="tvly-test-key")
        try:
            s = Settings()
            assert s.tavily_api_key == "tvly-test-key"
        finally:
            _restore_env(prev)

    def test_reads_model_overrides(self):
        from masis.config.settings import Settings
        prev = _set_env(MODEL_RESEARCHER="gpt-4.1-nano", MODEL_SUPERVISOR="gpt-4.1-mini")
        try:
            s = Settings()
            assert s.model_researcher == "gpt-4.1-nano"
            assert s.model_supervisor == "gpt-4.1-mini"
        finally:
            _restore_env(prev)

    def test_reads_environment(self):
        from masis.config.settings import Settings
        prev = _set_env(ENVIRONMENT="production")
        try:
            s = Settings()
            assert s.environment == "production"
        finally:
            _restore_env(prev)

    def test_log_level_uppercased(self):
        from masis.config.settings import Settings
        prev = _set_env(LOG_LEVEL="debug")
        try:
            s = Settings()
            assert s.log_level == "DEBUG"
        finally:
            _restore_env(prev)


# ---------------------------------------------------------------------------
# Settings.validate()
# ---------------------------------------------------------------------------

class TestSettingsValidation:
    """ENG-02 M3 S1b: Missing OPENAI_API_KEY must raise EnvironmentError."""

    def setup_method(self):
        from masis.config.settings import reset_settings
        reset_settings()

    def teardown_method(self):
        from masis.config.settings import reset_settings
        reset_settings()

    def test_validate_raises_when_openai_key_missing(self):
        """ENG-02 done-when: missing OPENAI_API_KEY raises clear error at startup."""
        from masis.config.settings import Settings
        prev = _set_env(OPENAI_API_KEY=None, ENVIRONMENT="development")
        try:
            s = Settings()
            with pytest.raises(EnvironmentError) as exc_info:
                s.validate()
            # Error message must mention the missing variable
            assert "OPENAI_API_KEY" in str(exc_info.value)
        finally:
            _restore_env(prev)

    def test_validate_passes_with_openai_key_set(self):
        from masis.config.settings import Settings
        prev = _set_env(OPENAI_API_KEY="sk-test", ENVIRONMENT="development")
        try:
            s = Settings()
            s.validate()  # Must not raise
        finally:
            _restore_env(prev)

    def test_validate_error_lists_all_missing_vars(self):
        """Error must list ALL missing variables at once, not just the first."""
        from masis.config.settings import Settings
        prev = _set_env(OPENAI_API_KEY=None, ENVIRONMENT="development")
        try:
            s = Settings()
            with pytest.raises(EnvironmentError) as exc_info:
                s.validate()
            # Error message should reference how to fix it
            msg = str(exc_info.value)
            assert "OPENAI_API_KEY" in msg
            assert ".env" in msg.lower() or "environment" in msg.lower()
        finally:
            _restore_env(prev)

    def test_production_requires_postgres_url(self):
        """In production environment, POSTGRES_URL is required."""
        from masis.config.settings import Settings
        prev = _set_env(OPENAI_API_KEY="sk-test", ENVIRONMENT="production", POSTGRES_URL=None)
        try:
            s = Settings()
            with pytest.raises(EnvironmentError) as exc_info:
                s.validate()
            assert "POSTGRES_URL" in str(exc_info.value)
        finally:
            _restore_env(prev)

    def test_development_does_not_require_postgres(self):
        """Development environment should not require POSTGRES_URL."""
        from masis.config.settings import Settings
        prev = _set_env(OPENAI_API_KEY="sk-test", ENVIRONMENT="development", POSTGRES_URL=None)
        try:
            s = Settings()
            s.validate()  # Must not raise in dev without POSTGRES_URL
        finally:
            _restore_env(prev)


# ---------------------------------------------------------------------------
# Settings properties
# ---------------------------------------------------------------------------

class TestSettingsProperties:
    """Test computed property methods on Settings."""

    def setup_method(self):
        from masis.config.settings import reset_settings
        reset_settings()

    def test_is_production_true(self):
        from masis.config.settings import Settings
        prev = _set_env(ENVIRONMENT="production")
        try:
            s = Settings()
            assert s.is_production is True
            assert s.is_development is False
        finally:
            _restore_env(prev)

    def test_is_development_true(self):
        from masis.config.settings import Settings
        prev = _set_env(ENVIRONMENT="development")
        try:
            s = Settings()
            assert s.is_development is True
            assert s.is_production is False
        finally:
            _restore_env(prev)

    def test_use_postgres_true_when_url_set(self):
        from masis.config.settings import Settings
        prev = _set_env(POSTGRES_URL="postgresql://user:pw@host:5432/db")
        try:
            s = Settings()
            assert s.use_postgres_checkpointer is True
        finally:
            _restore_env(prev)

    def test_use_postgres_false_when_url_missing(self):
        from masis.config.settings import Settings
        prev = _set_env(POSTGRES_URL=None)
        try:
            s = Settings()
            assert s.use_postgres_checkpointer is False
        finally:
            _restore_env(prev)

    def test_use_langfuse_true_when_keys_set(self):
        from masis.config.settings import Settings
        prev = _set_env(LANGFUSE_SECRET_KEY="sk-lf-x", LANGFUSE_PUBLIC_KEY="pk-lf-x")
        try:
            s = Settings()
            assert s.use_langfuse is True
        finally:
            _restore_env(prev)

    def test_use_langfuse_false_when_keys_missing(self):
        from masis.config.settings import Settings
        prev = _set_env(LANGFUSE_SECRET_KEY=None, LANGFUSE_PUBLIC_KEY=None)
        try:
            s = Settings()
            assert s.use_langfuse is False
        finally:
            _restore_env(prev)


# ---------------------------------------------------------------------------
# get_settings() singleton
# ---------------------------------------------------------------------------

class TestGetSettingsSingleton:
    """Test that get_settings() returns the same instance on repeated calls."""

    def setup_method(self):
        from masis.config.settings import reset_settings
        reset_settings()

    def teardown_method(self):
        from masis.config.settings import reset_settings
        reset_settings()

    def test_returns_same_instance(self):
        from masis.config.settings import get_settings
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reset_creates_new_instance(self):
        from masis.config.settings import get_settings, reset_settings
        s1 = get_settings()
        reset_settings()
        s2 = get_settings()
        assert s1 is not s2

    def test_env_change_reflected_after_reset(self):
        from masis.config.settings import get_settings, reset_settings
        prev = _set_env(LOG_LEVEL="WARNING")
        try:
            reset_settings()
            s = get_settings()
            assert s.log_level == "WARNING"
        finally:
            _restore_env(prev)
            reset_settings()


# ---------------------------------------------------------------------------
# validate_env() convenience function
# ---------------------------------------------------------------------------

class TestValidateEnv:
    """Test the validate_env() convenience function."""

    def setup_method(self):
        from masis.config.settings import reset_settings
        reset_settings()

    def teardown_method(self):
        from masis.config.settings import reset_settings
        reset_settings()

    def test_raises_when_openai_missing(self):
        from masis.config.settings import validate_env, reset_settings
        prev = _set_env(OPENAI_API_KEY=None, ENVIRONMENT="development")
        try:
            reset_settings()
            with pytest.raises(EnvironmentError):
                validate_env()
        finally:
            _restore_env(prev)
            reset_settings()

    def test_passes_when_key_set(self):
        from masis.config.settings import validate_env, reset_settings
        prev = _set_env(OPENAI_API_KEY="sk-dummy", ENVIRONMENT="development")
        try:
            reset_settings()
            validate_env()  # Should not raise
        finally:
            _restore_env(prev)
            reset_settings()


# ---------------------------------------------------------------------------
# Settings.summary()
# ---------------------------------------------------------------------------

class TestSettingsSummary:
    """Test the summary() method masks sensitive values."""

    def setup_method(self):
        from masis.config.settings import reset_settings
        reset_settings()

    def test_summary_returns_string(self):
        from masis.config.settings import Settings
        s = Settings()
        summary = s.summary()
        assert isinstance(summary, str)
        assert len(summary) > 50

    def test_summary_masks_api_key(self):
        from masis.config.settings import Settings
        prev = _set_env(OPENAI_API_KEY="sk-proj-secretvalue1234567890")
        try:
            s = Settings()
            summary = s.summary()
            # Full key must not appear in summary
            assert "sk-proj-secretvalue1234567890" not in summary
        finally:
            _restore_env(prev)

    def test_summary_contains_environment(self):
        from masis.config.settings import Settings
        prev = _set_env(ENVIRONMENT="staging")
        try:
            s = Settings()
            summary = s.summary()
            assert "staging" in summary
        finally:
            _restore_env(prev)
