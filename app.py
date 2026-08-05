import base64
import json
import os
import re
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
    title="Gemini Enterprise Multi-Role Agent Middleware",
    description="Enterprise Production Middleware with RBAC for Sales and Engineering Teams",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registry of A2A Agents by Role / Department
AGENTS_REGISTRY = {
    "sales": {
        "id": "10498432126001584255",
        "name": "Gemini Enterprise Sales & Infographic Agent",
        "description": "ผู้ช่วยทีมขาย สรุปข้อมูลฟีเจอร์ และสร้างภาพ 3D Infographic",
        "url": (
            "https://discoveryengine.googleapis.com/v1/projects/241553676885/locations/global"
            "/collections/default_collection/engines/tms-gemini-enterprise_1781074214544"
            "/assistants/default_assistant/agents/10498432126001584255/a2a/v1/message:send"
        ),
    },
    "engineering": {
        "id": "15088006752941300029",
        "name": "Foundation Reinforcement BOQ Extraction Agent",
        "description": "AI Quantity Surveyor สำหรับถอดปริมาณเหล็กเสริมและวิเคราะห์แบบฐานราก",
        "url": (
            "https://discoveryengine.googleapis.com/v1/projects/241553676885/locations/global"
            "/collections/default_collection/engines/tms-gemini-enterprise_1781074214544"
            "/assistants/default_assistant/agents/15088006752941300029/a2a/v1/message:send"
        ),
    },
}

ACCESS_TOKEN_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
API_KEY_SECRET = os.getenv("API_KEY_SECRET", "sales-team-secret-key-2026")
DEFAULT_GCP_TOKEN = os.getenv("DEFAULT_GCP_TOKEN")


def sanitize_token(token: Optional[str]) -> str:
    """Remove newlines, carriage returns, and accidental spaces from OAuth token."""
    if not token:
        return ""
    return token.replace("\n", "").replace("\r", "").replace(" ", "").strip()


def get_access_token(incoming_auth: Optional[str] = None) -> str:
    """Fetch OAuth access token via Incoming Header, ENV token, SA Key ENV, google-auth (ADC), or gcloud CLI."""
    if incoming_auth and incoming_auth.startswith("Bearer "):
        token = incoming_auth.replace("Bearer ", "").strip()
        if token and token != API_KEY_SECRET:
            return sanitize_token(token)

    if DEFAULT_GCP_TOKEN and DEFAULT_GCP_TOKEN.strip():
        return sanitize_token(DEFAULT_GCP_TOKEN)

    sa_json = os.getenv("GCP_SERVICE_ACCOUNT_KEY") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if sa_json:
        try:
            import google.auth.transport.requests
            from google.oauth2 import service_account

            clean_json = sa_json.strip()
            if not clean_json.startswith("{"):
                clean_json = base64.b64decode(clean_json).decode("utf-8")

            info = json.loads(clean_json)
            creds = service_account.Credentials.from_service_account_info(
                info, scopes=ACCESS_TOKEN_SCOPES
            )
            auth_req = google.auth.transport.requests.Request()
            creds.refresh(auth_req)
            return sanitize_token(creds.token)
        except Exception as e:
            print(f"Error loading SA credentials from ENV: {e}")

    try:
        import google.auth
        import google.auth.transport.requests
        credentials, _ = google.auth.default(scopes=ACCESS_TOKEN_SCOPES)
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        return sanitize_token(credentials.token)
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token", f"--scopes={ACCESS_TOKEN_SCOPES[0]}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return sanitize_token(result.stdout)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to obtain GCP access token. Details: {str(e)}",
        )


def extract_text(data: dict) -> str:
    """Extract text parts AND generated images (inlineData, fileData, artifacts) from A2A response payload."""
    chunks = []

    def collect_from_message(msg: dict) -> None:
        for part in msg.get("content") or msg.get("parts") or []:
            if "text" in part:
                chunks.append(part["text"])
            elif "inlineData" in part:
                inline = part["inlineData"]
                mime = inline.get("mimeType", "image/png")
                b64_data = inline.get("data", "")
                if b64_data:
                    chunks.append(f"\n\n![Generated Image](data:{mime};base64,{b64_data})\n\n")
            elif "fileData" in part:
                fdata = part["fileData"]
                uri = fdata.get("fileUri") or fdata.get("uri") or ""
                if uri:
                    chunks.append(f"\n\n![Generated Image]({uri})\n\n")
            elif "uri" in part:
                chunks.append(f"\n\n![Generated Image]({part['uri']})\n\n")

    if isinstance(data.get("message"), dict):
        collect_from_message(data["message"])
    if data.get("content") or data.get("parts"):
        collect_from_message(data)

    status_msg = data.get("status", {}).get("message")
    if status_msg:
        collect_from_message(status_msg)
    for artifact in data.get("artifacts", []) or []:
        if isinstance(artifact, dict):
            collect_from_message(artifact)
            art_uri = artifact.get("uri") or artifact.get("url") or artifact.get("gcsPath")
            if art_uri and not any(art_uri in c for c in chunks):
                chunks.append(f"\n\n![Artifact Image]({art_uri})\n\n")
    for hist_msg in data.get("history", []) or []:
        if isinstance(hist_msg, dict):
            collect_from_message(hist_msg)

    return "".join(chunks)


class ChatRequest(BaseModel):
    message: str
    role: Optional[str] = "sales"  # Default role is sales ('sales' or 'engineering')
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    file_base64: Optional[str] = None
    mime_type: Optional[str] = None
    file_name: Optional[str] = None
    access_token: Optional[str] = None


class ChatResponse(BaseModel):
    status: str
    role: str
    agent_name: str
    reply: str
    session_id: str
    user_id: Optional[str] = None
    file_attached: Optional[str] = None
    raw_response: Optional[dict] = None


@app.post("/api/chat", response_model=ChatResponse)
def chat_with_agent(req: ChatRequest, authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
    if x_api_key and x_api_key != API_KEY_SECRET:
        raise HTTPException(status_code=401, detail="Invalid X-API-KEY header.")

    if not req.message.strip() and not req.file_base64:
        raise HTTPException(status_code=400, detail="Message or file attachment is required.")

    # Determine targeted Agent by Role
    target_role = (req.role or "sales").lower()
    if target_role not in AGENTS_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Invalid role '{target_role}'. Allowed roles: {list(AGENTS_REGISTRY.keys())}")

    agent_info = AGENTS_REGISTRY[target_role]
    send_url = agent_info["url"]

    session_id = req.session_id or uuid.uuid4().hex
    user_id = req.user_id or f"user_{target_role}"

    token = sanitize_token(req.access_token) or get_access_token(authorization)

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
            send_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
    except requests.RequestException as e:
        err_msg = str(e)
        err_msg = re.sub(r"ya29\.[A-Za-z0-9_\-]+", "[REDACTED_TOKEN]", err_msg)
        raise HTTPException(
            status_code=502, detail=f"Failed to connect to Agent API: {err_msg}"
        )

    if not res.ok:
        err_msg = re.sub(r"ya29\.[A-Za-z0-9_\-]+", "[REDACTED_TOKEN]", res.text)
        raise HTTPException(
            status_code=res.status_code,
            detail=f"Agent API Error ({res.status_code}): {err_msg}",
        )

    data = res.json()
    reply_text = extract_text(data)

    if not reply_text.strip():
        reply_text = "(Agent returned empty response)"

    return ChatResponse(
        status="success",
        role=target_role,
        agent_name=agent_info["name"],
        reply=reply_text,
        session_id=session_id,
        user_id=user_id,
        file_attached=req.file_name,
        raw_response=data,
    )


@app.get("/api/agents")
def get_available_agents():
    """Return available agents by department role for frontend UI."""
    return {
        "status": "ok",
        "roles": list(AGENTS_REGISTRY.keys()),
        "agents": [
            {
                "role_key": k,
                "name": v["name"],
                "description": v["description"],
                "id": v["id"]
            }
            for k, v in AGENTS_REGISTRY.items()
        ]
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "Gemini Enterprise Multi-Role Agent Proxy",
        "active_roles": list(AGENTS_REGISTRY.keys()),
        "features": ["multi_role_rbac", "text", "file_upload", "image_extraction", "line_webhook"]
    }


@app.get("/", response_class=HTMLResponse)
def render_ui():
    with open("/Users/chatchawand./ge_a2a_spike/static/index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
