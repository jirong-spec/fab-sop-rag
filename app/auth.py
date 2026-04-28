"""
Optional API key authentication.

Behaviour:
  - If API_KEY is set in config, every request to protected endpoints must
    include a matching X-API-Key header.
  - If API_KEY is empty (default), authentication is disabled and all requests
    are allowed — suitable for internal demos and PoC environments.

Future: replace with JWT / OAuth2 middleware for production deployments.
"""

import logging
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from app.config import settings

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """FastAPI dependency: validate X-API-Key when API_KEY is configured."""
    if not settings.api_key:
        return  # auth disabled
    if api_key != settings.api_key:
        logger.warning("Rejected request with invalid or missing API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key (X-API-Key header required)",
        )
