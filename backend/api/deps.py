"""
Auth dependencies for write endpoints.

Checks X-Admin-Token header against ADMIN_TOKEN env var.
"""
from __future__ import annotations

from fastapi import Header, HTTPException

from backend.config import Config


async def require_admin(x_admin_token: str = Header(default="")) -> str:
    """FastAPI dependency — validates admin token. Returns actor string."""
    if not Config.ADMIN_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="ADMIN_TOKEN not configured on server — write endpoints disabled",
        )
    if x_admin_token != Config.ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing X-Admin-Token")
    return "admin"
