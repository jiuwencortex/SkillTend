# coding: utf-8
"""Example 5: Session-resume counter hydration.

The Problem
-----------
On gateway platforms (Telegram, Discord, web chat), Jiuwen creates a FRESH
agent instance for every incoming message.  A fresh `BackgroundReviewRail`
starts with both nudge counters at zero.

Without hydration, this means:
  - Session already has 9 user turns (counter should be at 9 of 10 threshold)
  - User sends the 10th message — fresh agent, fresh rail → counter = 1
  - Review NEVER fires because the counter never reaches the threshold

The Fix: hydrate_from_history()
--------------------------------
Mirrors Hermes gateway session hydration in conversation_loop.py:

    prior_user_turns = sum(1 for m in history if m["role"] == "user")
    agent._turns_since_memory = prior_user_turns % memory_nudge_interval

Uses modulo to restore the ORIGINAL cadence without immediately firing a
review on the first resumed message.

Usage:
    python examples/agent_evolving_hermess/05_session_resume_hydration.py
"""
import asyncio
from agent_evolving.online import BackgroundReviewConfig, BackgroundReviewRail


def simulate_history(num_user_turns: int):
    """Build a fake conversation history with num_user_turns user messages."""
    messages = []
    for i in range(num_user_turns):
        messages.append({"role": "user", "content": f"User message {i + 1}"})
        messages.append({"role": "assistant", "content": f"Assistant reply {i + 1}"})
    return messages


async def main():
    config = BackgroundReviewConfig(
        enabled=True,
        skill_nudge_interval=10,
        memory_nudge_interval=10,
        flush_min_turns=6,
        review_model="gpt-4o-mini",
    )

    print("── Scenario 1: Fresh session (no history) ──────────────────")
    rail_fresh = BackgroundReviewRail(config=config)
    print(f"  Counters (fresh)  : {rail_fresh.pending_counts()}")
    # user_turns_since_review=0, tool_iters=0

    print("\n── Scenario 2: Gateway resume — 9 prior turns ──────────────")
    history_9 = simulate_history(9)
    rail_gateway = BackgroundReviewRail(config=config)
    rail_gateway.hydrate_from_history(history_9)
    counts = rail_gateway.pending_counts()
    print(f"  Prior user turns  : 9")
    print(f"  Nudge interval    : {config.memory_nudge_interval}")
    print(f"  Counters after hydration: {counts}")
    # user_turns_since_review = 9 % 10 = 9  ← one turn away from firing
    assert counts["user_turns_since_review"] == 9 % config.memory_nudge_interval
    print("  ✓ Counter correctly restored to 9")

    print("\n── Scenario 3: Gateway resume — 25 prior turns ─────────────")
    history_25 = simulate_history(25)
    rail_25 = BackgroundReviewRail(config=config)
    rail_25.hydrate_from_history(history_25)
    counts = rail_25.pending_counts()
    print(f"  Prior user turns  : 25")
    print(f"  Counters after hydration: {counts}")
    # user_turns_since_review = 25 % 10 = 5  ← mid-cycle
    assert counts["user_turns_since_review"] == 25 % config.memory_nudge_interval
    print("  ✓ Counter correctly restored to 5 (mid-cycle)")

    print("\n── Scenario 4: Idempotent — hydrating twice ─────────────────")
    rail_idempotent = BackgroundReviewRail(config=config)
    rail_idempotent.hydrate_from_history(history_9)
    rail_idempotent.hydrate_from_history(history_25)  # second call is a no-op
    counts = rail_idempotent.pending_counts()
    assert counts["user_turns_since_review"] == 9 % config.memory_nudge_interval
    print("  ✓ Second hydrate_from_history() is ignored (session already active)")

    print("\nAll assertions passed.")


if __name__ == "__main__":
    asyncio.run(main())
