# coding: utf-8
"""Example 2: Online background review — custom config and manual trigger simulation.

Demonstrates:
  - Custom nudge thresholds (fire every 3 tool calls in testing)
  - Protected skill names (immutable skills)
  - Direct call to run_background_review() for testing/integration
  - Inspecting ReviewResult actions

Usage:
    python examples/agent_evolving_hermess/02_online_review_custom_config.py
"""
import asyncio
from pathlib import Path

from agent_evolving.online import (
    BackgroundReviewConfig,
    BackgroundReviewRail,
    ReviewMode,
    ReviewTrigger,
    run_background_review,
)


async def main():
    # ── 1. Custom config: low thresholds for fast testing ────────────────────
    config = BackgroundReviewConfig(
        enabled=True,
        skill_nudge_interval=3,      # trigger after just 3 tool calls (for testing)
        memory_nudge_interval=3,     # trigger after just 3 user turns
        review_model="gpt-4o-mini",
        review_timeout_seconds=30.0,
        review_max_iterations=8,
        # Protect bundled or system skills from being auto-edited
        protected_skill_names=["bundled-core", "system-prompt", "system-instructions"],
        skills_root=Path.home() / ".jiuwen" / "skills",
        memory_root=Path.home() / ".jiuwen" / "memories",
    )

    rail = BackgroundReviewRail(config=config)
    print(f"Rail created. Protected skills: {config.protected_skill_names}")

    # ── 2. Directly call run_background_review() for testing ─────────────────
    # Simulate a conversation that revealed a user preference and a skill issue.
    fake_messages = [
        {"role": "user", "content": "Please stop using bullet points in your answers."},
        {
            "role": "assistant",
            "content": (
                "Understood. I'll avoid bullet points going forward and write "
                "in flowing prose instead."
            ),
        },
        {"role": "user", "content": "Also, when doing git reviews, always check the test files."},
        {
            "role": "assistant",
            "content": "Noted — I'll always inspect test files during git reviews.",
        },
    ]

    trigger = ReviewTrigger(
        mode=ReviewMode.COMBINED,   # review both memory AND skills
        user_turn_count=2,
        tool_iter_count=0,
        session_id="test-session-001",
    )

    print("\nRunning background review directly (requires litellm + API key)...")
    print("If no API key is configured, run_background_review returns gracefully with error.")

    try:
        result = await run_background_review(
            messages_snapshot=fake_messages,
            trigger=trigger,
            config=config,
            model=config.review_model or "gpt-4o-mini",
            session_id="test-session-001",
        )
    except Exception as exc:
        print(f"Review call raised: {exc}")
        return

    # ── 3. Inspect result ────────────────────────────────────────────────────
    print(f"\n── ReviewResult ──────────────────────────────────")
    print(f"  Summary    : {result.summary_line}")
    print(f"  Duration   : {result.duration_seconds:.2f}s")
    print(f"  Error      : {result.error or 'none'}")
    print(f"  Actions    : {len(result.actions)}")
    for action in result.actions:
        print(f"    [{action.action_type}] {action.target_name} — {action.summary}")


if __name__ == "__main__":
    asyncio.run(main())
