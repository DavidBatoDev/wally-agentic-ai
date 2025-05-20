# backend/src/dependencies/__init__.py
"""
Dependency injection utilities for FastAPI.
"""

from .auth import get_current_user
from .agent import get_agent_orchestrator

__all__ = [
    "get_current_user",
    "get_agent_orchestrator",
]