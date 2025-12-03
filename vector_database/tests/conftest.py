import os
import time

import pytest
import requests
from dotenv import load_dotenv

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant_test")
QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")
QDRANT_API_KEY = os.getenv("QDRANT_TEST_API_KEY", "test_key")


def wait_for_qdrant(timeout: int = 30):
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections"
    headers = {}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY

    start_time = time.time()
    while True:
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return
        except requests.exceptions.ConnectionError:
            pass

        if time.time() - start_time > timeout:
            raise RuntimeError("Qdrant does not respond")
        time.sleep(0.5)


@pytest.fixture(scope="session", autouse=True)
def ensure_qdrant_ready():
    wait_for_qdrant()
