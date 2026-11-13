"""Tests for FastAPI endpoints and concurrent request safety."""

import asyncio

import httpx

import backend.main as backend_main
from inference.engine import InferenceOutput


class FakeEngine:
    def __init__(self, loaded: bool = True) -> None:
        self.is_loaded = loaded
        self.model_variant = "instruction-finetuned"
        self.device = "cuda"
        self.context_length = 1024
        self.active_checkpoint_path = "checkpoints/gpt2_instruct_124m.pth"
        self.load_error = None if loaded else "Checkpoint missing"

    def load(self) -> None:
        self.is_loaded = True

    def generate_reply(
        self,
        message: str,
        history=None,
        temperature: float = 0.7,
        top_k: int = 40,
        max_tokens: int = 256,
        auto_length: bool = True,
    ) -> InferenceOutput:
        return InferenceOutput(
            response=f"Echo: {message}",
            model_variant=self.model_variant,
            device=self.device,
            tokens_generated=4,
            token_budget=min(max_tokens, 96),
            length_category="short" if auto_length else "custom",
            stopped_reason="eos",
            generation_time_ms=12.5,
        )


def test_health_and_chat_endpoints(monkeypatch) -> None:
    fake = FakeEngine(loaded=True)
    monkeypatch.setattr(backend_main, "engine", fake)

    async def _run() -> None:
        transport = httpx.ASGITransport(app=backend_main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/api/health")
            assert health.status_code == 200
            h_json = health.json()
            assert h_json["status"] == "ok"
            assert h_json["model_loaded"] is True
            assert h_json["model_variant"] == "instruction-finetuned"

            chat = await client.post(
                "/api/chat",
                json={
                    "message": "What is the capital of Japan?",
                    "temperature": 0.7,
                    "top_k": 40,
                    "max_tokens": 256,
                    "auto_length": True,
                },
            )
            assert chat.status_code == 200
            c_json = chat.json()
            assert c_json["response"] == "Echo: What is the capital of Japan?"
            assert c_json["stopped_reason"] == "eos"

            blank = await client.post("/api/chat", json={"message": "   "})
            assert blank.status_code == 422

    asyncio.run(_run())


def test_chat_returns_503_when_model_not_loaded(monkeypatch) -> None:
    fake = FakeEngine(loaded=False)
    monkeypatch.setattr(backend_main, "engine", fake)

    async def _run() -> None:
        transport = httpx.ASGITransport(app=backend_main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.post("/api/chat", json={"message": "Hello"})
            assert res.status_code == 503

    asyncio.run(_run())


def test_concurrent_chat_requests(monkeypatch) -> None:
    fake = FakeEngine(loaded=True)
    monkeypatch.setattr(backend_main, "engine", fake)

    async def _run() -> None:
        transport = httpx.ASGITransport(app=backend_main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            responses = await asyncio.gather(
                *[
                    client.post("/api/chat", json={"message": f"Question {i}"})
                    for i in range(8)
                ]
            )
            for i, res in enumerate(responses):
                assert res.status_code == 200
                assert res.json()["response"] == f"Echo: Question {i}"

    asyncio.run(_run())
