"""Dynamic token-budget selection based on prompt intent and explicit length cues."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenBudgetDecision:
    """Computed token budget and rationale for a prompt."""

    token_budget: int
    length_category: str  # "brief" | "short" | "moderate" | "detailed" | "custom"
    reason: str


_BRIEF_PATTERNS = (
    r"\bbriefly\b",
    r"\bin brief\b",
    r"\bshort answer\b",
    r"\bin short\b",
    r"\bone sentence\b",
    r"\bsingle sentence\b",
    r"\bone line\b",
    r"\bin a few words\b",
    r"\bconcise(?:ly)?\b",
    r"\btl;?dr\b",
    r"\bsummarize\b",
    r"\bquick answer\b",
    r"\byes or no\b",
)

_DETAILED_PATTERNS = (
    r"\bin detail\b",
    r"\bdetailed\b",
    r"\bstep[\s-]by[\s-]step\b",
    r"\bcomprehensive\b",
    r"\bthorough(?:ly)?\b",
    r"\bdeep dive\b",
    r"\bin depth\b",
    r"\bexplain fully\b",
    r"\bwalk me through\b",
    r"\bcomplete guide\b",
    r"\btutorial\b",
    r"\bwith examples\b",
    r"\bcompare and contrast\b",
    r"\bpros and cons\b",
)

_SIMPLE_FACTUAL_STARTERS = (
    "what is ",
    "what's ",
    "who is ",
    "who was ",
    "when did ",
    "when was ",
    "where is ",
    "define ",
    "convert ",
    "translate ",
    "spell ",
    "correct ",
    "rewrite ",
    "identify ",
    "what are the first ",
    "what color ",
    "how many ",
    "is ",
    "are ",
    "can you name ",
)

_GREETING_EXACT = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "bye",
}


def select_token_budget(
    prompt: str,
    user_max_tokens: int = 256,
    auto_length: bool = True,
    context_length: int = 1024,
    prompt_token_count: int = 0,
) -> TokenBudgetDecision:
    """Choose an appropriate max_new_tokens ceiling based on prompt complexity and cues.

    `token_budget` is an upper bound (not a target length): the model stops early
    whenever it emits `<|endoftext|>`.
    """
    available_context = max(16, context_length - prompt_token_count - 1)
    ceiling = min(max(16, int(user_max_tokens)), available_context)

    if not auto_length:
        return TokenBudgetDecision(
            token_budget=ceiling,
            length_category="custom",
            reason="Dynamic length disabled; using configured max_tokens limit.",
        )

    normalized = " ".join(prompt.lower().strip().split())
    word_count = len(normalized.split())

    # 1. Explicit brevity instruction takes highest priority
    if any(re.search(pattern, normalized) for pattern in _BRIEF_PATTERNS):
        budget = min(72, ceiling)
        return TokenBudgetDecision(
            token_budget=budget,
            length_category="brief",
            reason="Explicit brevity cue detected in prompt.",
        )

    # 2. Explicit detailed or step-by-step instruction
    if any(re.search(pattern, normalized) for pattern in _DETAILED_PATTERNS):
        # Allow generous ceiling up to 384 (or user ceiling if higher/lower)
        target_cap = max(ceiling, min(384, available_context)) if user_max_tokens >= 256 else ceiling
        budget = min(target_cap, available_context)
        return TokenBudgetDecision(
            token_budget=budget,
            length_category="detailed",
            reason="Explicit detailed or step-by-step cue detected in prompt.",
        )

    # 3. Greetings or ultra-short prompts
    cleaned_alpha = re.sub(r"[^\w\s]", "", normalized)
    if cleaned_alpha in _GREETING_EXACT or word_count <= 3:
        budget = min(64, ceiling)
        return TokenBudgetDecision(
            token_budget=budget,
            length_category="short",
            reason="Short greeting or minimal prompt detected.",
        )

    # 4. Simple factual / transformation questions
    if word_count <= 14 and any(normalized.startswith(s) for s in _SIMPLE_FACTUAL_STARTERS):
        budget = min(96, ceiling)
        return TokenBudgetDecision(
            token_budget=budget,
            length_category="short",
            reason="Simple factual or direct transformation query detected.",
        )

    # 5. Multi-part or longer prompts get a larger budget
    if word_count >= 25 or normalized.count("?") >= 2 or "explain how" in normalized or "why does" in normalized:
        budget = min(max(192, min(280, ceiling)), ceiling)
        return TokenBudgetDecision(
            token_budget=budget,
            length_category="detailed",
            reason="Complex or multi-part explanation request detected.",
        )

    # 6. General question -> moderate budget
    budget = min(160, ceiling)
    return TokenBudgetDecision(
        token_budget=budget,
        length_category="moderate",
        reason="General question detected; using moderate token budget.",
    )

