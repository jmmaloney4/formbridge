"""Pytest configuration for FormBridge tests."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require real PDFs)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )


# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)
