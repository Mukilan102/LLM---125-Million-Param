"""Tests for dynamic token-budget selection."""

from backend.token_budget import select_token_budget


def test_brief_prompt_gets_short_budget() -> None:
    decision = select_token_budget("Explain briefly what a neural network is.", user_max_tokens=256)
    assert decision.length_category == "brief"
    assert decision.token_budget <= 72


def test_simple_factual_question_gets_short_budget() -> None:
    decision = select_token_budget("What is the capital of France?", user_max_tokens=256)
    assert decision.length_category == "short"
    assert decision.token_budget <= 96


def test_general_question_gets_moderate_budget() -> None:
    decision = select_token_budget(
        "How do software engineers keep distributed services reliable under heavy traffic?",
        user_max_tokens=256,
    )
    assert decision.length_category == "moderate"
    assert 96 < decision.token_budget <= 160


def test_detailed_prompt_gets_larger_budget() -> None:
    decision = select_token_budget(
        "Explain step by step how backpropagation updates neural network weights.",
        user_max_tokens=256,
    )
    assert decision.length_category == "detailed"
    assert decision.token_budget >= 256


def test_auto_length_disabled_respects_user_max_tokens() -> None:
    decision = select_token_budget(
        "Explain briefly what an API is.",
        user_max_tokens=180,
        auto_length=False,
    )
    assert decision.length_category == "custom"
    assert decision.token_budget == 180


def test_budget_respects_context_window() -> None:
    decision = select_token_budget(
        "Explain in detail how transformers work.",
        user_max_tokens=512,
        auto_length=True,
        context_length=1024,
        prompt_token_count=960,
    )
    assert decision.token_budget <= 63

