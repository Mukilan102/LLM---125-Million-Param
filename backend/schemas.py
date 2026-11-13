"""Pydantic schemas for FastAPI chat and health endpoints."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    """Single conversation turn."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    """Request payload for POST /api/chat."""

    message: str = Field(..., min_length=1, max_length=4000, description="User prompt text")
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Optional prior conversation messages for multi-turn context",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 for greedy decoding)",
    )
    top_k: Optional[int] = Field(
        default=40,
        ge=1,
        le=500,
        description="Top-K sampling cutoff",
    )
    max_tokens: int = Field(
        default=256,
        ge=16,
        le=512,
        description="Upper limit on generated tokens",
    )
    auto_length: bool = Field(
        default=True,
        description="Enable dynamic token-budget selection based on prompt complexity",
    )

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Message must not be empty or whitespace only.")
        return stripped


class ChatResponse(BaseModel):
    """Response payload for POST /api/chat."""

    response: str
    model_variant: str
    device: str
    tokens_generated: int
    token_budget: int
    length_category: str
    stopped_reason: str
    generation_time_ms: float


class HealthResponse(BaseModel):
    """Response payload for GET /api/health."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_variant: str
    device: str
    context_length: int
    checkpoint_path: str
    error: Optional[str] = None

