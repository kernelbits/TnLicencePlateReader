"""Test setup for the backend API.

backend/backend.py creates a Roboflow client at import time and talks to
Supabase inside its endpoints. Both SDKs are replaced with stand-ins here, and
the environment is pinned so a developer's local .env cannot switch on real
clients. Individual tests mock the remaining HTTP calls (PaddleOCR, Ollama).
"""
import os
import sys
import types
from unittest import mock

import pytest

# load_dotenv() does not override variables that are already set, so pinning
# them here keeps the import offline and deterministic.
os.environ["SUPABASE_URL"] = ""
os.environ["SUPABASE_ANON_KEY"] = ""
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = ""
os.environ["RF_API_KEY"] = "test-key"
os.environ["OCR_API_URL"] = "http://ocr.test/layout-parsing"
os.environ["OCR_TOKEN"] = "test-token"
os.environ["OLLAMA_API_URL"] = "http://ollama.test/api/generate"

_roboflow = types.ModuleType("roboflow")
_roboflow.Roboflow = mock.MagicMock(name="Roboflow")
sys.modules["roboflow"] = _roboflow

_supabase = types.ModuleType("supabase")
_supabase.create_client = mock.MagicMock(name="create_client")
_supabase.Client = object
sys.modules["supabase"] = _supabase


@pytest.fixture
def backend_module():
    import backend.backend as module

    module.model.reset_mock()
    module.supabase = None
    return module


@pytest.fixture
def client(backend_module):
    from fastapi.testclient import TestClient

    return TestClient(backend_module.app)
