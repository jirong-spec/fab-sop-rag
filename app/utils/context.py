"""
Request-scoped context variables.

Centralised here so that both the middleware and the logging filter
can import from one place without circular dependencies.
"""

import contextvars

# Set by RequestIDMiddleware at the start of every request.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


def get_request_id() -> str:
    return request_id_var.get("-")
