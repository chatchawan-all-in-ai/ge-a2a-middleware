# /// script
# requires-python = ">=3.9"
# dependencies = ["requests"]
# ///
"""Talk to the Creator Agent over A2A (HTTP+JSON).

The agent card advertises:
    url:               .../a2a          (A2A base)
    preferredTransport: HTTP+JSON        (REST binding -> POST {url}/v1/message:send)

Auth: bearer token from `gcloud auth print-access-token`.

Usage:
    uv run talk_to_creator.py "your message here"
    uv run talk_to_creator.py            # falls back to the default prompt below
"""

import subprocess
import sys
import uuid

import requests

# A2A base URL from the agent card.
AGENT_BASE_URL = (
    "https://discoveryengine.googleapis.com/v1/projects/241553676885/locations/global"
    "/collections/default_collection/engines/tms-gemini-enterprise_1781074214544"
    "/assistants/default_assistant/agents/10498432126001584255/a2a"
)


SEND_URL = f"{AGENT_BASE_URL}/v1/message:send"

DEFAULT_PROMPT = "Hello. What you can do?"

# OAuth scope required to call the Discovery Engine / Gemini Enterprise A2A API.
ACCESS_TOKEN_SCOPES = "https://www.googleapis.com/auth/cloud-platform"


def get_access_token() -> str:
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token", f"--scopes={ACCESS_TOKEN_SCOPES}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def build_payload(text: str) -> dict:
    """Google a2a.v1 message:send request body.

    Note the Google-specific shape: message is wrapped in `request`, `role`
    is an enum (ROLE_USER), and parts are under `content`.
    """
    return {
        "request": {
            "role": "ROLE_USER",
            "content": [{"text": text}],
            "messageId": uuid.uuid4().hex,
        }
    }


def extract_text(data: dict) -> str:
    """Pull text parts out of an A2A Message or Task response."""
    chunks = []

    def collect_from_message(msg: dict) -> None:
        # Generic A2A uses "parts"; Google a2a.v1 uses "content".
        for part in (msg.get("content") or msg.get("parts") or []):
            if "text" in part:
                chunks.append(part["text"])

    # Google a2a.v1 wraps the reply in a top-level "message".
    if isinstance(data.get("message"), dict):
        collect_from_message(data["message"])

    # Response can be a bare Message...
    if data.get("content") or data.get("parts"):
        collect_from_message(data)

    # ...or a Task with status.message and/or artifacts.
    status_msg = data.get("status", {}).get("message")
    if status_msg:
        collect_from_message(status_msg)
    for artifact in data.get("artifacts", []) or []:
        collect_from_message(artifact)
    for hist_msg in data.get("history", []) or []:
        collect_from_message(hist_msg)

    # Content parts are streaming fragments split mid-sentence -> concatenate.
    return "".join(chunks)


def _cwidth(ch: str) -> int:
    """Display columns for a single char (2 for wide/emoji, 0 for combining)."""
    import unicodedata

    if unicodedata.combining(ch):
        return 0
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return 2
    # Common emoji / symbol blocks render double-width.
    if any(a <= ord(ch) <= b for a, b in (
        (0x1F300, 0x1FAFF), (0x2600, 0x27BF), (0x2B00, 0x2BFF), (0xFE00, 0xFE0F),
    )):
        return 2
    return 1


def _swidth(s: str) -> int:
    return sum(_cwidth(ch) for ch in s)


def _wrap(text: str, width: int) -> list:
    """Wrap text to a max display width, preserving explicit line breaks."""
    lines = []
    for raw in text.splitlines() or [""]:
        if not raw.strip():
            lines.append("")
            continue
        cur, cur_w = "", 0
        for word in raw.split(" "):
            ww = _swidth(word)
            add = ww + (1 if cur else 0)
            if cur and cur_w + add > width:
                lines.append(cur)
                cur, cur_w = word, ww
            else:
                cur = f"{cur} {word}" if cur else word
                cur_w += add
        lines.append(cur)
    return lines


def render_bubble(speaker: str, text: str, *, align_right: bool) -> None:
    """Print a single chat bubble aligned left (agent) or right (you)."""
    term_width = 80
    body_width = min(60, term_width - 4) - 4  # room for "│ " and " │"

    lines = _wrap(text.strip(), body_width)
    inner = max([_swidth(line) for line in lines] + [_swidth(speaker)], default=0)

    pad = (term_width - (inner + 4)) if align_right else 0
    lead = " " * max(pad, 0)

    print(f"{lead}┌─ {speaker} " + "─" * max(inner - _swidth(speaker) - 1, 0) + "┐")
    for line in lines:
        fill = " " * (inner - _swidth(line))
        print(f"{lead}│ {line}{fill} │")
    print(f"{lead}└" + "─" * (inner + 2) + "┘")


def main() -> int:
    prompt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT
    token = get_access_token()

    render_bubble("You", prompt, align_right=True)
    print()

    response = requests.post(
        SEND_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=build_payload(prompt),
        timeout=120,
    )

    try:
        data = response.json()
    except ValueError:
        render_bubble("Agent", response.text.strip() or "(empty response)", align_right=False)
        return 0 if response.ok else 1

    if response.ok:
        text = extract_text(data)
        if text.strip():
            render_bubble("Agent", text, align_right=False)
            return 0

    # Concise error — no full JSON dump.
    err = data.get("error", {}).get("message") if isinstance(data, dict) else None
    render_bubble("Agent", err or f"(no text in response, HTTP {response.status_code})", align_right=False)
    return 0 if response.ok else 1


if __name__ == "__main__":
    sys.exit(main())
