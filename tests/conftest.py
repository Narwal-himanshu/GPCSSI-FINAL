from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def client():
    from src.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def access_log_lines():
    path = ROOT / "examplelog" / "my_access.log.txt"
    return path.read_text(encoding="utf-8").splitlines()


def parse_sse(text: str) -> list[dict]:
    import json

    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]
