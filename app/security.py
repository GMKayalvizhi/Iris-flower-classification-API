"""
API-key authentication for protected routes.

Uses FastAPI's Security/Depends system with a custom header (X-API-Key).
Applied at the router level (see app/routers/v1.py, v2.py) so every
endpoint under a protected router is covered automatically -- no
per-endpoint decorator to forget.
"""

import logging
import secrets
from typing import Optional

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from app.config import settings

logger = logging.getLogger("iris_api")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(
    request: Request,
    api_key: Optional[str] = Security(api_key_header),
) -> None:
    """
    Raise 401 if the X-API-Key header is missing or doesn't match the
    configured key. Uses a constant-time comparison to avoid leaking
    timing information about how much of the key is correct.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    if api_key is None or not secrets.compare_digest(api_key, settings.API_KEY):
        logger.warning(
            f"Rejected request: missing or invalid API key request_id={request_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
        )