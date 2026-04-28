"""
Request ID middleware.

- Reads X-Request-ID from the incoming request header, or generates a short UUID.
- Stores it in a contextvars.ContextVar so every log line in the request
  lifecycle can include it automatically via RequestIDFilter.
- Echoes the ID back in the X-Request-ID response header.
"""

import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

from app.utils.context import request_id_var

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        # Accept a client-supplied ID (useful for end-to-end tracing) or mint a new one.
        req_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())[:8]
        token = request_id_var.set(req_id)
        request.state.request_id = req_id
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers[REQUEST_ID_HEADER] = req_id
        return response
