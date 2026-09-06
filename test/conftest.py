import os
from uuid import uuid4

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires test Docker services")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.get_closest_marker("legacy_e2e") and os.getenv("RUN_LEGACY_E2E") != "1":
            item.add_marker(pytest.mark.skip(reason="set RUN_LEGACY_E2E=1 with API/LLM services for legacy smoke tests"))
        elif item.get_closest_marker("integration") and os.getenv("RUN_INTEGRATION_TESTS") != "1":
            item.add_marker(pytest.mark.skip(reason="set RUN_INTEGRATION_TESTS=1 after docker compose up"))


@pytest.fixture
def unique_username():
    return f"itest_{uuid4().hex[:12]}"
