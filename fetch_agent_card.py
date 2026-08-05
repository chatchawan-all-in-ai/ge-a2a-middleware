# /// script
# requires-python = ">=3.9"
# dependencies = ["requests"]
# ///
"""Fetch and print the A2A agent card from Discovery Engine.

Auth: uses the access token from `gcloud auth print-access-token`.

Run with:
    uv run fetch_agent_card.py
"""

import subprocess
import sys

import requests

URL = (
    "https://discoveryengine.googleapis.com/v1/projects/241553676885/locations/global"
    "/collections/default_collection/engines/tms-gemini-enterprise_1781074214544"
    "/assistants/default_assistant/agents/15088006752941300029/a2a/v1/card"
)


def get_access_token() -> str:
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def main() -> int:
    token = get_access_token()
    response = requests.get(
        URL,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    print(f"HTTP {response.status_code}\n")
    print(response.text)
    return 0 if response.ok else 1


if __name__ == "__main__":
    sys.exit(main())
