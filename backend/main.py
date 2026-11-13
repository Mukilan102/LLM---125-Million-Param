"""FastAPI application exposing GPT-2 chat and health endpoints."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from backend.config import (
    CORS_ORIGINS,
    INSTRUCT_CHECKPOINT_PATH,
    PRETRAINED_CHECKPOINT_PATH,
    TF_MODELS_DIR,
)
from backend.schemas import ChatRequest, ChatResponse, HealthResponse
from inference.engine import GPT2InferenceEngine

logger = logging.getLogger("gpt2_chat_backend")
logging.basicConfig(level=logging.INFO)

engine = GPT2InferenceEngine(
    instruct_checkpoint_path=INSTRUCT_CHECKPOINT_PATH,
    pretrained_checkpoint_path=PRETRAINED_CHECKPOINT_PATH,
    tf_models_dir=TF_MODELS_DIR,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load GPT-2 model once at backend startup."""
    try:
        logger.info("Loading GPT-2 model onto device=%s ...", engine.device)
        engine.load()
        logger.info(
            "Loaded GPT-2 (%s) from %s",
            engine.model_variant,
            engine.active_checkpoint_path,
        )
    except Exception as exc:
        logger.exception("Failed to load GPT-2 model during startup: %s", exc)
    yield


app = FastAPI(
    title="Custom GPT-2 Chat API",
    description="Local GPT-2 124M chat backend powered by PyTorch and FastAPI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return backend and model readiness status."""
    return HealthResponse(
        status="ok" if engine.is_loaded else "degraded",
        model_loaded=engine.is_loaded,
        model_variant=engine.model_variant,
        device=str(engine.device),
        context_length=engine.context_length,
        checkpoint_path=engine.active_checkpoint_path,
        error=engine.load_error,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """Generate a response from the local GPT-2 model."""
    if not engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not ready: {engine.load_error or 'not loaded'}",
        )

    history_dicts = (
        [turn.model_dump() for turn in payload.history]
        if payload.history
        else None
    )

    try:
        result = await run_in_threadpool(
            engine.generate_reply,
            message=payload.message,
            history=history_dicts,
            temperature=payload.temperature,
            top_k=payload.top_k,
            max_tokens=payload.max_tokens,
            auto_length=payload.auto_length,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference error during /api/chat: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An internal inference error occurred while generating the response.",
        ) from exc

    return ChatResponse(
        response=result.response,
        model_variant=result.model_variant,
        device=result.device,
        tokens_generated=result.tokens_generated,
        token_budget=result.token_budget,
        length_category=result.length_category,
        stopped_reason=result.stopped_reason,
        generation_time_ms=result.generation_time_ms,
    )

