"""
masis.api
=========
Phase 4 FastAPI application package for MASIS (ENG-15).

Exposes the MASIS REST API with five endpoints:

    POST /masis/query           -- Start a new query (MF-API-01)
    POST /masis/resume          -- Resume from HITL pause (MF-API-02)
    GET  /masis/status/{id}     -- Check query status (MF-API-03)
    GET  /masis/trace/{id}      -- Full audit trail (MF-API-04)
    GET  /masis/stream/{id}     -- SSE event stream (MF-API-05)

Modules
-------
models  -- Pydantic request/response models for the API layer
main    -- FastAPI application with all endpoint handlers

Architecture reference
----------------------
final_architecture_and_flow.md Section 23.10
engineering_tasks.md ENG-15
"""

from masis.api.models import (
    QueryRequest,
    QueryResponse,
    ResumeRequest,
    StatusResponse,
    TraceResponse,
)
from masis.api.main import create_app

__all__ = [
    "QueryRequest",
    "QueryResponse",
    "ResumeRequest",
    "StatusResponse",
    "TraceResponse",
    "create_app",
]
