# coding: utf-8
"""Example 1: Online background review — basic usage.

Mirrors Hermes's background review daemon thread pattern.

What this does:
  After every 10 tool calls (skill_nudge_interval=10) OR
  every 10 user turns (memory_nudge_interval=10), the rail
  spawns an asyncio background task that:
    1. Reads the full conversation snapshot.
    2. Calls an LLM with review_prompt to detect skill/memory changes.
    3. Writes directly to ~/.jiuwen/skills/ and ~/.jiuwen/memories/.

Usage:
    python examples/agent_evolving_hermess/01_online_review_basic.py
"""
import asyncio
from pathlib import Path

from agent_evolving.online import BackgroundReviewConfig, BackgroundReviewRail


async def main():
    # ── 1. Create the rail with default configuration ─────────────────────────
    # Defaults mirror Hermes: 10-turn memory nudge, 10-tool skill nudge.
    config = BackgroundReviewConfig(
        enabled=True,
        skill_nudge_interval=10,     # review skills every 10 tool calls
        memory_nudge_interval=10,    # review memory every 10 user turns
        review_model="gpt-4o-mini",  # model for background review LLM
        # skills_root defaults to ~/.jiuwen/skills/
        # memory_root defaults to ~/.jiuwen/memories/
    )

    rail = BackgroundReviewRail(config=config)

    print("BackgroundReviewRail created.")
    print(f"  Priority : {rail.priority}  (runs after SkillEvolutionRail=80)")
    print(f"  Config   : skill_nudge={config.skill_nudge_interval}  "
          f"memory_nudge={config.memory_nudge_interval}")

    # ── 2. Register on a DeepAgent ────────────────────────────────────────────
    # In real code, you would do:
    #   agent = DeepAgent(...)
    #   await agent.register_rail(rail)
    #
    # The rail hooks into:
    #   after_model_call  → increments _user_turn_count
    #   after_tool_call   → increments _tool_iter_count
    #   after_invoke      → spawns background review task when thresholds hit

    # ── 3. Inspect pending counters ──────────────────────────────────────────
    counts = rail.pending_counts()
    print(f"\nPending counters: {counts}")
    # {"user_turns_since_review": 0, "tool_iters_since_review": 0}

    # ── 4. Inspect last result (None until first review fires) ───────────────
    result = rail.last_review_result()
    print(f"Last review result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
