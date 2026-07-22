# coding: utf-8
"""Example 6: Memory context block + skills system prompt injection.

Two new helpers that mirror Hermes's system prompt assembly:

1.  MemoryStore.build_memory_context_block()
    → Wraps MEMORY.md + USER.md entries in <memory-context> fences
    → Ready to inject into the VOLATILE tier of the agent system prompt

2.  build_skills_system_prompt(skills_root)
    → Builds a compact skill index from all SKILL.md files
    → Ready to inject into the STABLE tier of the agent system prompt

Hermes injects both into the system prompt so the model has:
  - Persistent recall of user preferences and past observations
  - An index of available skills to invoke

Usage:
    python examples/agent_evolving_hermess/06_memory_context_and_skills_prompt.py
"""
import asyncio
import tempfile
from pathlib import Path

from agent_evolving.online import MemoryStore, build_skills_system_prompt


async def demo_memory_context_block():
    """Show build_memory_context_block() output."""
    print("── MemoryStore.build_memory_context_block() ─────────────────")

    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(memory_root=Path(tmpdir))

        # Add some demo entries
        store.add("memory", "User prefers short, direct answers with no bullet points.")
        store.add("memory", "User is working on the Jiuwen swarm orchestration project.")
        store.add("user", "Name: Alex. Role: Senior ML Engineer. Language: Python only.")
        store.add("user", "Timezone: UTC+3. Works 9am-6pm local time.")

        block = store.build_memory_context_block()
        print(block)
        print()

        # Empty store returns empty string
        empty_store = MemoryStore(memory_root=Path(tmpdir) / "empty")
        assert empty_store.build_memory_context_block() == ""
        print("  ✓ Empty store returns empty string (no noisy injection)")

        # char_counts utility
        counts = store.char_counts()
        print(f"\n  Character usage: memory={counts['memory']} user={counts['user']}")


async def demo_skills_system_prompt():
    """Show build_skills_system_prompt() output."""
    print("\n── build_skills_system_prompt() ─────────────────────────────")

    with tempfile.TemporaryDirectory() as tmpdir:
        skills_root = Path(tmpdir)

        # Create two demo skills
        skill1 = skills_root / "git-review"
        skill1.mkdir()
        (skill1 / "SKILL.md").write_text(
            "---\nname: git-review\ndescription: Guidelines for reviewing git diffs and PRs.\n---\n\n"
            "When reviewing a PR: check logic, tests, security.\n",
            encoding="utf-8",
        )

        skill2 = skills_root / "coding" / "systematic-debugging"
        skill2.mkdir(parents=True)
        (skill2 / "SKILL.md").write_text(
            "---\nname: systematic-debugging\n"
            "description: Step-by-step methodology for diagnosing bugs.\n"
            "immutable: true\n---\n\nUse binary search...\n",
            encoding="utf-8",
        )

        prompt = await build_skills_system_prompt(skills_root)
        print(prompt)
        print()

        # Empty skills root returns empty string
        empty_prompt = await build_skills_system_prompt(Path(tmpdir) / "no-skills")
        assert empty_prompt == ""
        print("  ✓ Empty skills root returns empty string")


if __name__ == "__main__":
    asyncio.run(demo_memory_context_block())
    asyncio.run(demo_skills_system_prompt())
