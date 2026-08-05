import os
import subprocess
import sys
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests

app = FastAPI(
    title="Gemini Enterprise Agent External Middleware",
    description="Enterprise Production Middleware for Sales Team to access Gemini Enterprise Agent via A2A Protocol",
    version="2.0.0",
)

# Enable CORS for GitHub Pages and all frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AGENT_BASE_URL = (
    "https://discoveryengine.googleapis.com/v1/projects/241553676885/locations/global"
    "/collections/default_collection/engines/tms-gemini-enterprise_1781074214544"
    "/assistants/default_assistant/agents/10498432126001584255/a2a"
)
SEND_URL = f"{AGENT_BASE_URL}/v1/message:send"
ACCESS_TOKEN_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
API_KEY_SECRET = os.getenv("API_KEY_SECRET", "sales-team-secret-key-2026")


def get_access_token() -> str:
    """Fetch OAuth access token via google-auth (ADC) or fallback to gcloud CLI."""
    try:
        import google.auth
        import google.auth.transport.requests
        credentials, _ = google.auth.default(scopes=ACCESS_TOKEN_SCOPES)
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        return credentials.token
    except Exception:
        # Fallback for local development using gcloud CLI
        try:
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token", f"--scopes={ACCESS_TOKEN_SCOPES[0]}"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to obtain access token: {e.stderr}"
            )


def extract_text(data: dict) -> str:
    """Extract text parts from A2A response payload."""
    chunks = []

    def collect_from_message(msg: dict) -> None:
        for part in msg.get("content") or msg.get("parts") or []:
            if "text" in part:
                chunks.append(part["text"])

    if isinstance(data.get("message"), dict):
        collect_from_message(data["message"])
    if data.get("content") or data.get("parts"):
        collect_from_message(data)

    status_msg = data.get("status", {}).get("message")
    if status_msg:
        collect_from_message(status_msg)
    for artifact in data.get("artifacts", []) or []:
        collect_from_message(artifact)
    for hist_msg in data.get("history", []) or []:
        collect_from_message(hist_msg)

    return "".join(chunks)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    file_base64: Optional[str] = None
    mime_type: Optional[str] = None
    file_name: Optional[str] = None


class ChatResponse(BaseModel):
    status: str
    reply: str
    session_id: str
    user_id: Optional[str] = None
    file_attached: Optional[str] = None
    raw_response: Optional[dict] = None


@app.post("/api/chat", response_model=ChatResponse)
def chat_with_agent(req: ChatRequest, x_api_key: Optional[str] = Header(None)):
    if x_api_key and x_api_key != API_KEY_SECRET:
        raise HTTPException(status_code=401, detail="Invalid X-API-KEY header.")

    if not req.message.strip() and not req.file_base64:
        raise HTTPException(status_code=400, detail="Message or file attachment is required.")

    session_id = req.session_id or uuid.uuid4().hex
    user_id = req.user_id or "sales_team_member"
    token = get_access_token()

    content_parts = []
    if req.message.strip():
        content_parts.append({"text": req.message})

    if req.file_base64 and req.mime_type:
        content_parts.append({
            "inlineData": {
                "mimeType": req.mime_type,
                "data": req.file_base64
            }
        })

    payload = {
        "request": {
            "role": "ROLE_USER",
            "content": content_parts,
            "messageId": session_id,
        }
    }

    try:
        res = requests.post(
            SEND_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502, detail=f"Failed to connect to Agent API: {str(e)}"
        )

    if not res.ok:
        raise HTTPException(
            status_code=res.status_code,
            detail=f"Agent API Error ({res.status_code}): {res.text}",
        )

    data = res.json()
    reply_text = extract_text(data)

    if not reply_text.strip():
        reply_text = "(Agent returned empty response)"

    return ChatResponse(
        status="success",
        reply=reply_text,
        session_id=session_id,
        user_id=user_id,
        file_attached=req.file_name,
        raw_response=data,
    )


@app.post("/api/webhook/line")
def line_webhook_handler(payload: dict):
    events = payload.get("events", [])
    if not events:
        return {"status": "ok", "message": "No events"}

    event = events[0]
    user_id = event.get("source", {}).get("userId", "line_user")
    user_msg = event.get("message", {}).get("text", "")

    if not user_msg:
        return {"status": "ok", "message": "Non-text message ignored"}

    token = get_access_token()
    a2a_payload = {
        "request": {
            "role": "ROLE_USER",
            "content": [{"text": user_msg}],
            "messageId": user_id,
        }
    }

    try:
        res = requests.post(
            SEND_URL,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=a2a_payload,
            timeout=120,
        )
        data = res.json()
        reply_text = extract_text(data)
        return {
            "status": "success",
            "line_user_id": user_id,
            "reply_for_line": reply_text
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "Gemini Enterprise A2A Proxy",
        "features": ["text", "file_upload", "cors_enabled", "api_key_auth", "line_webhook", "adc_auth"]
    }


@app.get("/", response_class=HTMLResponse)
def render_ui():
    with open("/Users/chatchawand./ge_a2a_spike/static/index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
