# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Curated evaluation task bank — 60 tasks across 6 domains.

Each Task captures the minimal information needed to:
  1. Generate a synthetic agent conversation (via SessionSimulator)
  2. Judge whether a review-produced skill is correct (ground_truth + skill_keywords)
  3. Reproduce experiments exactly (all randomness is seeded outside the task)

Domains (10 tasks each):
  code_debug     — diagnosing and fixing code defects
  doc_draft      — writing technical documentation and structured text
  data_analysis  — transforming, aggregating, and interpreting data
  api_integration — calling external services, handling auth and errors
  sys_admin      — operating system configuration and diagnostics
  qa             — explaining technical concepts and comparing options
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Task:
    """One evaluation task."""

    id: str
    """Unique identifier, e.g. 'code_debug_01'."""

    domain: str
    """One of: code_debug, doc_draft, data_analysis, api_integration, sys_admin, qa."""

    difficulty: str
    """easy / medium / hard."""

    title: str
    """Short human-readable name."""

    user_query: str
    """The user's opening message to the agent."""

    solution_steps: List[str]
    """Ordered list of reasoning/action steps the agent takes.
    The SessionSimulator turns these into assistant + tool messages."""

    ground_truth: str
    """The correct final answer or outcome.  Used by the LLM judge in TSR scoring."""

    skill_keywords: List[str]
    """Keywords that a high-quality skill derived from this task should contain.
    Used as a soft signal in Retrieval Relevance scoring."""

    tool_names: List[str] = field(default_factory=list)
    """Names of tools the agent calls while solving this task (for conversation realism)."""


class TaskBank:
    """Indexed collection of Task objects."""

    def __init__(self, tasks: List[Task]) -> None:
        self._by_id: Dict[str, Task] = {t.id: t for t in tasks}
        self._tasks = tasks

    def get(self, task_id: str) -> Task:
        if task_id not in self._by_id:
            raise KeyError(f"Unknown task_id: {task_id!r}")
        return self._by_id[task_id]

    def by_domain(self, domain: str) -> List[Task]:
        return [t for t in self._tasks if t.domain == domain]

    def by_difficulty(self, difficulty: str) -> List[Task]:
        return [t for t in self._tasks if t.difficulty == difficulty]

    def all(self) -> List[Task]:
        return list(self._tasks)

    def ids(self) -> List[str]:
        return [t.id for t in self._tasks]

    @property
    def domains(self) -> List[str]:
        return sorted({t.domain for t in self._tasks})


# ── Domain: code_debug ────────────────────────────────────────────────────────

_CODE_DEBUG: List[Task] = [
    Task(
        id="code_debug_01",
        domain="code_debug",
        difficulty="easy",
        title="KeyError in dict access",
        user_query=(
            "My Python script crashes with KeyError: 'user_id' when processing API responses. "
            "Here is the offending line: `name = response['data']['user_id']`. "
            "The key is definitely in some responses but not all. How do I fix this?"
        ),
        solution_steps=[
            "Explain that KeyError occurs when the key is absent; dict.get() returns None instead of raising.",
            "Show the fix: use `.get()` with a default or check with `in` before accessing.",
            "Recommend defensive pattern: `user_id = response.get('data', {}).get('user_id')` for nested access.",
            "Suggest adding a log statement when the key is absent to detect upstream API changes.",
        ],
        ground_truth=(
            "Use `response.get('data', {}).get('user_id')` for safe nested access. "
            "This returns None instead of raising KeyError when any level is missing."
        ),
        skill_keywords=["KeyError", "dict.get", "nested access", "safe lookup", "default value"],
        tool_names=["code_edit"],
    ),
    Task(
        id="code_debug_02",
        domain="code_debug",
        difficulty="medium",
        title="Async race condition — shared mutable state",
        user_query=(
            "I have two asyncio tasks that both append to the same list. "
            "Occasionally items are duplicated or lost. "
            "The list is a module-level variable. How do I fix the race condition?"
        ),
        solution_steps=[
            "Explain that asyncio is single-threaded but tasks interleave at await points.",
            "Show that list.append() is atomic within a single event loop iteration — duplication/loss "
            "suggests the bug is in the read-modify-write pattern (e.g., `lst = lst + [item]`).",
            "Fix: use asyncio.Lock around any compound read-modify-write on shared state.",
            "Show example: `async with lock: shared_list.append(item)`.",
            "Recommend replacing shared mutable list with an asyncio.Queue for producer-consumer patterns.",
        ],
        ground_truth=(
            "Protect compound read-modify-write with asyncio.Lock. "
            "For producer-consumer patterns prefer asyncio.Queue over a shared list."
        ),
        skill_keywords=["asyncio", "race condition", "Lock", "shared state", "Queue", "atomic"],
        tool_names=["code_edit", "bash"],
    ),
    Task(
        id="code_debug_03",
        domain="code_debug",
        difficulty="medium",
        title="SQL query returns empty result unexpectedly",
        user_query=(
            "My PostgreSQL query `SELECT * FROM orders WHERE status = 'pending'` returns 0 rows, "
            "but I can see pending orders in pgAdmin. The table has 1,200 rows. "
            "I'm using SQLAlchemy with a connection pool."
        ),
        solution_steps=[
            "First hypothesis: the query runs in a different transaction that doesn't see uncommitted rows.",
            "Instruct user to check if they called session.commit() after inserting the pending orders.",
            "Second hypothesis: the status column has trailing whitespace — suggest `TRIM(status) = 'pending'`.",
            "Third hypothesis: case sensitivity — PostgreSQL is case-sensitive by default.",
            "Recommend: print the raw SQL and parameters via `echo=True` on the engine to confirm the query.",
        ],
        ground_truth=(
            "Most likely cause: rows inserted in an uncommitted transaction. "
            "Call session.commit() or check isolation level. "
            "Also verify no trailing whitespace in the status column."
        ),
        skill_keywords=["SQLAlchemy", "transaction", "commit", "isolation level", "PostgreSQL", "whitespace"],
        tool_names=["bash", "code_edit"],
    ),
    Task(
        id="code_debug_04",
        domain="code_debug",
        difficulty="hard",
        title="Memory leak in long-running Python process",
        user_query=(
            "Our Python microservice's RSS memory grows by ~50 MB per hour under normal load "
            "and never shrinks. We use FastAPI, SQLAlchemy, and Redis. "
            "The process runs for days before OOM-killing. How do I find the leak?"
        ),
        solution_steps=[
            "Use tracemalloc to take memory snapshots at two points in time and diff them.",
            "Show how to enable tracemalloc at startup and dump the top 10 allocations.",
            "Common culprits: SQLAlchemy sessions not closed (use contextmanager), "
            "Redis connections not returned to pool, large objects held in module-level caches.",
            "Recommend using `objgraph.show_growth()` to see which object types are growing.",
            "Add a `/debug/memory` health endpoint that dumps tracemalloc stats in production.",
        ],
        ground_truth=(
            "Enable tracemalloc at startup; compare snapshots with tracemalloc.take_snapshot(). "
            "Most common leak: SQLAlchemy sessions not closed — use `with Session() as s:`. "
            "Use objgraph for live object-count inspection."
        ),
        skill_keywords=["tracemalloc", "memory leak", "objgraph", "SQLAlchemy session", "contextmanager"],
        tool_names=["bash", "code_edit"],
    ),
    Task(
        id="code_debug_05",
        domain="code_debug",
        difficulty="easy",
        title="Off-by-one error in loop boundary",
        user_query=(
            "My function is supposed to process elements at indices 0 through n-1 but it "
            "always skips the last element. The loop is `for i in range(1, n):`. What is wrong?"
        ),
        solution_steps=[
            "Explain: range(1, n) produces 1,2,...,n-1 — it both starts too late (skips index 0) "
            "and would end at n-1 which in 0-indexed terms skips the last element if n is the length.",
            "Fix: `for i in range(n):` or `for i in range(0, n):` to include index 0.",
            "Show the fencepost rule: range(start, stop) includes start, excludes stop.",
        ],
        ground_truth="Change `range(1, n)` to `range(n)` to start from index 0 inclusive.",
        skill_keywords=["range", "off-by-one", "fencepost", "loop boundary", "0-indexed"],
        tool_names=["code_edit"],
    ),
    Task(
        id="code_debug_06",
        domain="code_debug",
        difficulty="medium",
        title="Regex not matching due to greedy quantifier",
        user_query=(
            "My regex `<.*>` is supposed to extract individual HTML tags from a string like "
            "`<b>hello</b><i>world</i>` but it matches the entire string as one group. "
            "Why and how do I fix it?"
        ),
        solution_steps=[
            "Explain greedy vs. lazy quantifiers: `.*` is greedy and matches as much as possible.",
            "Fix: use `.*?` (lazy) to match as little as possible: `<.*?>`.",
            "Better alternative: use a proper HTML parser (BeautifulSoup) for production HTML parsing.",
            "Show a test case in the Python re module to confirm.",
        ],
        ground_truth=(
            "Use `<.*?>` with lazy quantifier `?` to stop at the first `>`. "
            "For real HTML parsing use BeautifulSoup, not regex."
        ),
        skill_keywords=["regex", "greedy", "lazy", "quantifier", ".*?", "BeautifulSoup"],
        tool_names=["bash"],
    ),
    Task(
        id="code_debug_07",
        domain="code_debug",
        difficulty="medium",
        title="Circular import in Python package",
        user_query=(
            "I get `ImportError: cannot import name 'UserService' from partially initialized module 'services'`. "
            "The error happens when `routes.py` imports from `services.py` and `services.py` imports from `models.py` "
            "which imports from `routes.py`. How do I break the cycle?"
        ),
        solution_steps=[
            "Explain Python module initialization: a circular import causes a module to be partially initialized "
            "when another module tries to import from it.",
            "Strategy 1: move the import inside the function where it is needed (lazy import).",
            "Strategy 2: restructure — move the shared type to a third module (`types.py`) that neither "
            "services.py nor routes.py imports from each other for.",
            "Strategy 3: use TYPE_CHECKING guard for type-hint-only imports.",
            "Recommend: circular imports are a structural smell — prefer restructuring over lazy imports.",
        ],
        ground_truth=(
            "Break the cycle by extracting shared types to a neutral `types.py` module. "
            "Use `from __future__ import annotations` + `TYPE_CHECKING` guard for type-hint-only imports."
        ),
        skill_keywords=["circular import", "TYPE_CHECKING", "lazy import", "module initialization", "restructure"],
        tool_names=["code_edit"],
    ),
    Task(
        id="code_debug_08",
        domain="code_debug",
        difficulty="easy",
        title="Flask route not matching — trailing slash",
        user_query=(
            "My Flask route `@app.route('/api/users')` works fine, "
            "but when I call `GET /api/users/` (with trailing slash) I get a 404. "
            "How do I handle both?"
        ),
        solution_steps=[
            "Explain Flask's strict_slashes: by default `/api/users` and `/api/users/` are different routes.",
            "Fix 1: set `strict_slashes=False` on the route decorator.",
            "Fix 2: set `app.url_map.strict_slashes = False` globally.",
            "Note: the canonical way is to decide on one form and use redirects — "
            "Flask automatically redirects `/api/users/` → `/api/users` for routes defined without trailing slash.",
        ],
        ground_truth=(
            "Add `strict_slashes=False` to the route decorator: "
            "`@app.route('/api/users', strict_slashes=False)`. "
            "Or disable globally: `app.url_map.strict_slashes = False`."
        ),
        skill_keywords=["Flask", "strict_slashes", "trailing slash", "404", "route", "redirect"],
        tool_names=["code_edit"],
    ),
    Task(
        id="code_debug_09",
        domain="code_debug",
        difficulty="medium",
        title="Pandas merge produces unexpected NaN rows",
        user_query=(
            "I'm merging two DataFrames with `df1.merge(df2, on='user_id')` but the result has "
            "many NaN values in columns from df2, even for user_ids that I can see in df2. "
            "df1 has 5,000 rows, df2 has 4,800 rows, result has 5,000 rows."
        ),
        solution_steps=[
            "The 5,000-row result with NaN indicates a left join: default merge is inner join "
            "which would drop non-matching rows. The user likely wants inner but got left-style result — "
            "actually with inner join the result would be ≤ 5000. NaN in right columns → left join behavior.",
            "Check: `df1.merge(df2, on='user_id', how='left')` gives NaN where df2 has no matching user_id.",
            "Diagnose: check for dtype mismatch — `df1['user_id'].dtype` vs `df2['user_id'].dtype`. "
            "If one is int and other is str, no rows will match.",
            "Fix dtype: `df2['user_id'] = df2['user_id'].astype(int)` then re-merge.",
            "Use `indicator=True` in the merge to see which rows matched.",
        ],
        ground_truth=(
            "NaN in merge output most likely means a dtype mismatch on the key column. "
            "Check `df1['user_id'].dtype == df2['user_id'].dtype`; cast to match. "
            "Use `indicator=True` to diagnose which rows did not match."
        ),
        skill_keywords=["pandas", "merge", "NaN", "dtype", "left join", "indicator", "key mismatch"],
        tool_names=["bash", "code_edit"],
    ),
    Task(
        id="code_debug_10",
        domain="code_debug",
        difficulty="hard",
        title="pytest fixture scope causes state leakage between tests",
        user_query=(
            "Some of my pytest tests fail only when run after a specific other test. "
            "They pass in isolation. I use a `session`-scoped fixture that creates a database "
            "connection and inserts seed data. What is wrong?"
        ),
        solution_steps=[
            "Explain pytest fixture scopes: session-scoped fixtures are shared across all tests in the session — "
            "if one test modifies the shared state, subsequent tests see the modified state.",
            "Root cause: the session-scoped DB fixture inserts seed data once; a test that deletes or "
            "modifies rows leaves the DB dirty for subsequent tests.",
            "Fix 1: downgrade fixture to function scope so each test gets a fresh DB state.",
            "Fix 2: wrap each test in a transaction and roll it back (use SQLAlchemy's `nested` or "
            "`savepoint` for this).",
            "Fix 3: if session scope is required for performance, use `autouse` fixture that truncates "
            "affected tables before each test.",
            "Recommend: use pytest-postgresql or similar for isolated test databases.",
        ],
        ground_truth=(
            "Session-scoped fixtures share state across tests. "
            "Fix by using function scope, or wrapping each test in a rolled-back transaction. "
            "Avoid mutating session-scoped fixtures inside individual tests."
        ),
        skill_keywords=["pytest", "fixture", "scope", "session", "state leakage", "transaction", "rollback"],
        tool_names=["bash", "code_edit"],
    ),
]

# ── Domain: doc_draft ─────────────────────────────────────────────────────────

_DOC_DRAFT: List[Task] = [
    Task(
        id="doc_draft_01",
        domain="doc_draft",
        difficulty="easy",
        title="Write a technical README for a CLI tool",
        user_query=(
            "Write a README for a Python CLI tool called `dbmigrate` that applies SQL migrations. "
            "It has three commands: `up`, `down`, and `status`. "
            "Users install it via pip and configure it with a `dbmigrate.toml` file."
        ),
        solution_steps=[
            "Open with a one-sentence project description.",
            "Add install section: `pip install dbmigrate`.",
            "Add configuration section showing a minimal `dbmigrate.toml` example.",
            "Add usage section with examples of all three commands.",
            "Add a short Contributing section.",
        ],
        ground_truth=(
            "README with: description, install, config (dbmigrate.toml), "
            "usage examples for up/down/status commands, contributing section."
        ),
        skill_keywords=["README", "CLI", "install", "configuration", "usage examples", "commands"],
        tool_names=["file_write"],
    ),
    Task(
        id="doc_draft_02",
        domain="doc_draft",
        difficulty="medium",
        title="Write an API reference for a REST endpoint",
        user_query=(
            "Write API reference documentation for a `POST /api/v1/orders` endpoint. "
            "It accepts JSON with fields: `product_id` (string, required), `quantity` (integer, required, min 1), "
            "`notes` (string, optional). Returns 201 with order object, or 400/422 on validation errors."
        ),
        solution_steps=[
            "Write endpoint title, method, and URL.",
            "Write request body table with field name, type, required/optional, description.",
            "Write response section: 201 Created with example order JSON, 400 Bad Request, 422 Unprocessable.",
            "Write a curl example.",
            "Note authentication requirement (Bearer token in Authorization header).",
        ],
        ground_truth=(
            "API doc with: endpoint URL+method, request body table, response codes and examples, "
            "curl example, authentication note."
        ),
        skill_keywords=["API reference", "endpoint", "request body", "response", "status codes", "curl example"],
        tool_names=["file_write"],
    ),
    Task(
        id="doc_draft_03",
        domain="doc_draft",
        difficulty="hard",
        title="Write a system design document for a notification service",
        user_query=(
            "Write a system design document for a notification service that sends email, SMS, and push "
            "notifications. It must handle 10,000 notifications/second at peak. "
            "Include: requirements, high-level architecture, data model, failure handling."
        ),
        solution_steps=[
            "Write functional requirements: multi-channel delivery, template engine, delivery tracking.",
            "Write non-functional requirements: 10k/s throughput, at-least-once delivery, < 5s latency.",
            "Design architecture: API gateway → message queue (Kafka) → channel workers (email/SMS/push) → delivery DB.",
            "Write data model: Notification(id, user_id, channel, template_id, status, created_at, sent_at).",
            "Write failure handling: dead-letter queue, exponential backoff retry, idempotency keys.",
        ],
        ground_truth=(
            "System design doc with: functional + non-functional requirements, Kafka-based queue architecture, "
            "notification data model, at-least-once delivery via DLQ and retry."
        ),
        skill_keywords=["system design", "Kafka", "notification", "at-least-once", "retry", "DLQ", "throughput"],
        tool_names=["file_write"],
    ),
    Task(
        id="doc_draft_04",
        domain="doc_draft",
        difficulty="medium",
        title="Write a postmortem for a database outage",
        user_query=(
            "Write a postmortem for a 45-minute production database outage caused by a connection pool "
            "exhaustion. The service recovered after restarting the application tier. "
            "Use the standard 5-section format."
        ),
        solution_steps=[
            "Write Summary: 45-min outage, root cause, impact.",
            "Write Timeline: detection, investigation steps, mitigation, resolution.",
            "Write Root Cause: connection pool set to 10; a slow query held connections; "
            "new requests queued until timeout.",
            "Write Impact: % of requests failed, affected users.",
            "Write Action Items: increase pool size, add pool monitoring alert, add query timeout.",
        ],
        ground_truth=(
            "Postmortem with: summary, timeline, root cause (connection pool exhaustion + slow query), "
            "impact, action items (pool size, monitoring, query timeout)."
        ),
        skill_keywords=["postmortem", "root cause", "timeline", "action items", "connection pool", "mitigation"],
        tool_names=["file_write"],
    ),
    Task(
        id="doc_draft_05",
        domain="doc_draft",
        difficulty="easy",
        title="Write a deployment runbook",
        user_query=(
            "Write a deployment runbook for releasing a new version of a Python web service to production. "
            "The service runs on Kubernetes. Steps should include: pre-deploy checks, deploy command, "
            "smoke test, and rollback procedure."
        ),
        solution_steps=[
            "Pre-deploy: confirm staging passed CI, check no active incidents, notify on-call.",
            "Deploy: `kubectl set image deployment/myapp myapp=myimage:v2.3.1`.",
            "Monitor: watch rollout with `kubectl rollout status deployment/myapp`.",
            "Smoke test: hit `/health` endpoint, check error rate in Grafana for 10 minutes.",
            "Rollback: `kubectl rollout undo deployment/myapp`.",
        ],
        ground_truth=(
            "Runbook with: pre-deploy checklist, kubectl deploy command, rollout monitoring, "
            "smoke test steps, rollback command."
        ),
        skill_keywords=["runbook", "Kubernetes", "deployment", "rollback", "smoke test", "kubectl rollout"],
        tool_names=["file_write"],
    ),
    Task(
        id="doc_draft_06",
        domain="doc_draft",
        difficulty="easy",
        title="Write a code review checklist",
        user_query=(
            "Create a code review checklist for a Python backend team. "
            "Cover: correctness, security, performance, readability, and tests."
        ),
        solution_steps=[
            "Correctness: logic is correct, edge cases handled, no off-by-one errors.",
            "Security: no hardcoded secrets, inputs validated, SQL uses parameterized queries.",
            "Performance: no N+1 queries, no blocking I/O in async path, cache considered.",
            "Readability: names are clear, functions are small, comments explain why not what.",
            "Tests: new behavior has tests, tests are isolated, no real network/DB calls in unit tests.",
        ],
        ground_truth=(
            "Checklist covering correctness, security (no secrets, parameterized SQL), "
            "performance (N+1, async), readability, and test coverage."
        ),
        skill_keywords=["code review", "checklist", "security", "N+1", "parameterized", "test isolation"],
        tool_names=["file_write"],
    ),
    Task(
        id="doc_draft_07",
        domain="doc_draft",
        difficulty="medium",
        title="Document a data schema in Markdown",
        user_query=(
            "Write schema documentation for a `payments` database table with columns: "
            "id (UUID PK), order_id (UUID FK → orders.id), amount_cents (INTEGER NOT NULL), "
            "currency (CHAR(3) NOT NULL), status (ENUM: pending/captured/refunded), created_at (TIMESTAMPTZ)."
        ),
        solution_steps=[
            "Write table name, purpose, and any important notes (e.g., amounts stored as cents).",
            "Write columns table: name, type, constraints, description.",
            "Write indexes section.",
            "Write relationships section: FK to orders.",
            "Write notes: amounts in integer cents to avoid floating-point issues.",
        ],
        ground_truth=(
            "Schema doc with: purpose, columns table with types and constraints, "
            "FK relationship to orders, note that amount is stored as integer cents."
        ),
        skill_keywords=["schema", "data model", "foreign key", "integer cents", "enum", "documentation"],
        tool_names=["file_write"],
    ),
    Task(
        id="doc_draft_08",
        domain="doc_draft",
        difficulty="medium",
        title="Write a Git branching strategy document",
        user_query=(
            "Document our Git branching strategy for a team of 8 engineers. "
            "We use GitHub, deploy to staging on every merge to `main`, and do weekly releases to production."
        ),
        solution_steps=[
            "Describe branch types: main (production), feature/* (one per task), hotfix/* (production patches).",
            "Describe workflow: branch from main → PR → code review (1 approval required) → merge.",
            "Staging: every merge to main triggers CI/CD to staging automatically.",
            "Production release: weekly, tag main with semver, deploy tagged version.",
            "Hotfix: branch from production tag → fix → PR to main → cherry-pick to release.",
        ],
        ground_truth=(
            "Git strategy doc with: branch types (main/feature/hotfix), PR/review process, "
            "auto-staging on merge to main, weekly tagged production releases, hotfix procedure."
        ),
        skill_keywords=["Git", "branching", "PR", "staging", "release", "hotfix", "semver", "cherry-pick"],
        tool_names=["file_write"],
    ),
    Task(
        id="doc_draft_09",
        domain="doc_draft",
        difficulty="easy",
        title="Write a meeting agenda for a sprint planning session",
        user_query=(
            "Write a 1-hour sprint planning agenda for a 5-person team starting a 2-week sprint. "
            "Include time slots for each section."
        ),
        solution_steps=[
            "0:00-0:05 — Open: confirm attendees, share screen.",
            "0:05-0:20 — Review sprint goal and capacity (account for holidays/PTO).",
            "0:20-0:45 — Backlog review: walk through top stories, estimate if needed.",
            "0:45-0:55 — Commit: agree on sprint scope, assign owners.",
            "0:55-1:00 — Wrap: confirm sprint goal statement, next standup time.",
        ],
        ground_truth=(
            "Sprint planning agenda with time slots: open (5m), capacity review (15m), "
            "backlog walkthrough (25m), commitment (10m), wrap (5m)."
        ),
        skill_keywords=["sprint planning", "agenda", "capacity", "backlog", "time slots", "sprint goal"],
        tool_names=["file_write"],
    ),
    Task(
        id="doc_draft_10",
        domain="doc_draft",
        difficulty="hard",
        title="Write an architectural decision record (ADR)",
        user_query=(
            "Write an ADR for choosing PostgreSQL over MongoDB for a new analytics service. "
            "The service stores structured event records, needs complex aggregation queries, "
            "and must support ad-hoc reporting. Expected data volume: 500M records/year."
        ),
        solution_steps=[
            "Write context: analytics service, structured events, complex aggregations, 500M records/year.",
            "Write decision: choose PostgreSQL with TimescaleDB extension.",
            "Write rationale: ACID guarantees, mature aggregation (window functions, CTEs), "
            "TimescaleDB compression (up to 95%), strong tooling ecosystem.",
            "Write alternatives considered: MongoDB (flexible schema but weak aggregation, no window functions), "
            "ClickHouse (excellent for analytics but operational complexity), BigQuery (managed but expensive).",
            "Write consequences: need to design schema carefully upfront; benefit from SQL familiarity.",
        ],
        ground_truth=(
            "ADR choosing PostgreSQL+TimescaleDB: strong aggregation via SQL, TimescaleDB compression, "
            "ACID guarantees. Alternatives: MongoDB (weak aggregation), ClickHouse (complex ops), BigQuery (cost)."
        ),
        skill_keywords=["ADR", "PostgreSQL", "TimescaleDB", "MongoDB", "aggregation", "window functions", "decision"],
        tool_names=["file_write"],
    ),
]

# ── Domain: data_analysis ─────────────────────────────────────────────────────

_DATA_ANALYSIS: List[Task] = [
    Task(
        id="data_analysis_01",
        domain="data_analysis",
        difficulty="easy",
        title="Aggregate time-series data by hour",
        user_query=(
            "I have a pandas DataFrame with columns `timestamp` (datetime) and `value` (float). "
            "How do I compute the hourly mean and count of values?"
        ),
        solution_steps=[
            "Set timestamp as the index: `df = df.set_index('timestamp')`.",
            "Use resample: `df.resample('h').agg({'value': ['mean', 'count']})`.",
            "Flatten MultiIndex columns if needed.",
        ],
        ground_truth="Use `df.set_index('timestamp').resample('h').agg({'value': ['mean', 'count']})`.",
        skill_keywords=["pandas", "resample", "time-series", "aggregation", "hourly", "agg"],
        tool_names=["bash"],
    ),
    Task(
        id="data_analysis_02",
        domain="data_analysis",
        difficulty="medium",
        title="Detect outliers using IQR method",
        user_query=(
            "I have a column `response_time_ms` in a DataFrame. "
            "How do I remove outliers using the interquartile range (IQR) method?"
        ),
        solution_steps=[
            "Compute Q1 and Q3: `Q1 = df['response_time_ms'].quantile(0.25)`.",
            "Compute IQR: `IQR = Q3 - Q1`.",
            "Filter: `df_clean = df[(df['response_time_ms'] >= Q1 - 1.5*IQR) & (df['response_time_ms'] <= Q3 + 1.5*IQR)]`.",
            "Note: multiplier 1.5 is standard; 3.0 removes only extreme outliers.",
        ],
        ground_truth=(
            "Filter with Q1-1.5*IQR lower bound and Q3+1.5*IQR upper bound. "
            "IQR = Q3 - Q1. Use multiplier 3.0 for extreme-only removal."
        ),
        skill_keywords=["IQR", "outlier", "quantile", "pandas", "filter", "interquartile"],
        tool_names=["bash"],
    ),
    Task(
        id="data_analysis_03",
        domain="data_analysis",
        difficulty="easy",
        title="Compute Pearson correlation matrix",
        user_query="How do I compute a correlation matrix for all numeric columns in a pandas DataFrame?",
        solution_steps=[
            "Use `df.corr()` for Pearson correlation (default).",
            "For Spearman rank correlation: `df.corr(method='spearman')`.",
            "Visualize with seaborn: `sns.heatmap(df.corr(), annot=True, fmt='.2f')`.",
        ],
        ground_truth="Use `df.corr()` for Pearson; `df.corr(method='spearman')` for rank correlation. Visualize with `sns.heatmap`.",
        skill_keywords=["correlation", "Pearson", "Spearman", "heatmap", "seaborn", "df.corr"],
        tool_names=["bash"],
    ),
    Task(
        id="data_analysis_04",
        domain="data_analysis",
        difficulty="medium",
        title="Train a simple linear regression and evaluate it",
        user_query=(
            "I have features X and labels y (both numpy arrays). "
            "How do I train a linear regression, compute RMSE, and check for overfitting?"
        ),
        solution_steps=[
            "Split: `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)`.",
            "Train: `from sklearn.linear_model import LinearRegression; model = LinearRegression().fit(X_train, y_train)`.",
            "Evaluate: `rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))`.",
            "Check overfitting: compare train RMSE vs test RMSE; large gap → overfitting.",
            "Also report R²: `model.score(X_test, y_test)`.",
        ],
        ground_truth=(
            "train_test_split → LinearRegression.fit → predict → compute RMSE and R². "
            "Overfitting = large train-test RMSE gap."
        ),
        skill_keywords=["linear regression", "RMSE", "R²", "train_test_split", "overfitting", "sklearn"],
        tool_names=["bash"],
    ),
    Task(
        id="data_analysis_05",
        domain="data_analysis",
        difficulty="easy",
        title="Compute and plot a distribution histogram",
        user_query="How do I plot a histogram of a pandas Series `df['latency_ms']` with 50 bins and a KDE overlay?",
        solution_steps=[
            "Use seaborn: `sns.histplot(df['latency_ms'], bins=50, kde=True)`.",
            "Or matplotlib: `df['latency_ms'].plot.hist(bins=50)` then overlay KDE separately.",
            "Label axes and add title.",
        ],
        ground_truth="Use `sns.histplot(df['latency_ms'], bins=50, kde=True)` for combined histogram + KDE.",
        skill_keywords=["histogram", "KDE", "seaborn", "histplot", "distribution", "bins"],
        tool_names=["bash"],
    ),
    Task(
        id="data_analysis_06",
        domain="data_analysis",
        difficulty="easy",
        title="Compute rolling moving average",
        user_query=(
            "I have a time-indexed pandas Series of daily sales. "
            "How do I compute a 7-day rolling average and a 30-day rolling average?"
        ),
        solution_steps=[
            "Use `.rolling(window=7).mean()` for 7-day average.",
            "Chain: `df['sales_7d'] = df['sales'].rolling(7).mean()`.",
            "Similarly: `df['sales_30d'] = df['sales'].rolling(30).mean()`.",
            "Note: first 6 rows of 7d average will be NaN (insufficient history).",
        ],
        ground_truth="Use `series.rolling(7).mean()` for 7-day and `.rolling(30).mean()` for 30-day. First N-1 rows are NaN.",
        skill_keywords=["rolling", "moving average", "pandas", "window", "time-series", "NaN"],
        tool_names=["bash"],
    ),
    Task(
        id="data_analysis_07",
        domain="data_analysis",
        difficulty="medium",
        title="Group-by and compute multiple aggregations",
        user_query=(
            "I have a DataFrame with `region`, `product`, and `revenue`. "
            "How do I compute total revenue, mean revenue, and order count per region+product combination?"
        ),
        solution_steps=[
            "Use `groupby` with `agg`: `df.groupby(['region', 'product']).agg(total=('revenue', 'sum'), "
            "mean=('revenue', 'mean'), count=('revenue', 'count'))`.",
            "Reset index to flatten: `.reset_index()`.",
            "Sort by total revenue descending: `.sort_values('total', ascending=False)`.",
        ],
        ground_truth=(
            "Use `.groupby(['region','product']).agg(total=('revenue','sum'), "
            "mean=('revenue','mean'), count=('revenue','count')).reset_index()`."
        ),
        skill_keywords=["groupby", "agg", "named aggregation", "MultiIndex", "reset_index", "sort_values"],
        tool_names=["bash"],
    ),
    Task(
        id="data_analysis_08",
        domain="data_analysis",
        difficulty="medium",
        title="Merge two DataFrames and handle mismatched keys",
        user_query=(
            "I'm merging `orders` and `customers` on `customer_id`. "
            "Some orders have customer_ids not in the customers table. "
            "How do I merge and find the unmatched orders?"
        ),
        solution_steps=[
            "Use left join: `merged = orders.merge(customers, on='customer_id', how='left', indicator=True)`.",
            "Find unmatched: `unmatched = merged[merged['_merge'] == 'left_only']`.",
            "Report: number and fraction of unmatched orders.",
            "Option: inner join to keep only matched; left join to keep all orders.",
        ],
        ground_truth=(
            "Left join with `indicator=True`, then filter `merged['_merge'] == 'left_only'` "
            "to find orders with no matching customer."
        ),
        skill_keywords=["merge", "left join", "indicator", "unmatched", "left_only", "customer_id"],
        tool_names=["bash"],
    ),
    Task(
        id="data_analysis_09",
        domain="data_analysis",
        difficulty="easy",
        title="Handle missing values — strategy selection",
        user_query=(
            "My DataFrame has NaN values in `age` (15% missing) and `city` (3% missing). "
            "What are my options for handling them and when should I use each?"
        ),
        solution_steps=[
            "Options: drop rows (`dropna`), fill with mean/median/mode (`fillna`), forward-fill for time-series, "
            "or impute with a model (sklearn SimpleImputer).",
            "For `age` (numeric, 15% missing): median imputation is safer than mean (robust to outliers). "
            "If MCAR (missing completely at random) consider row drop.",
            "For `city` (categorical, 3% missing): mode imputation or a separate 'Unknown' category.",
            "Never impute before train/test split (data leakage).",
        ],
        ground_truth=(
            "Numeric: median imputation (robust to outliers). Categorical: mode or 'Unknown' category. "
            "Impute only on training data; apply same transform to test data."
        ),
        skill_keywords=["missing values", "NaN", "imputation", "median", "fillna", "data leakage", "MCAR"],
        tool_names=["bash"],
    ),
    Task(
        id="data_analysis_10",
        domain="data_analysis",
        difficulty="medium",
        title="Filter, sort, and export a subset of data",
        user_query=(
            "I need to: (1) filter orders where status='completed' AND total > 100, "
            "(2) sort by created_at descending, (3) keep only columns order_id, customer_id, total, "
            "(4) export to CSV without the DataFrame index."
        ),
        solution_steps=[
            "Filter: `filtered = df[(df['status'] == 'completed') & (df['total'] > 100)]`.",
            "Sort: `sorted_df = filtered.sort_values('created_at', ascending=False)`.",
            "Select columns: `result = sorted_df[['order_id', 'customer_id', 'total']]`.",
            "Export: `result.to_csv('output.csv', index=False)`.",
        ],
        ground_truth=(
            "Chain: boolean filter with `&` → `.sort_values(ascending=False)` → column selection → `.to_csv(index=False)`."
        ),
        skill_keywords=["filter", "boolean indexing", "sort_values", "column selection", "to_csv", "index=False"],
        tool_names=["bash", "file_write"],
    ),
]

# ── Domain: api_integration ───────────────────────────────────────────────────

_API_INTEGRATION: List[Task] = [
    Task(
        id="api_integration_01",
        domain="api_integration",
        difficulty="hard",
        title="Implement OAuth 2.0 client credentials flow",
        user_query=(
            "I need to call an API that uses OAuth 2.0 client credentials flow. "
            "I have a client_id and client_secret. How do I get an access token and "
            "refresh it automatically when it expires?"
        ),
        solution_steps=[
            "POST to token endpoint with client_id, client_secret, grant_type=client_credentials.",
            "Parse response: access_token, expires_in.",
            "Cache the token and track expiry: refresh when `time.time() > token_expiry - 60`.",
            "Show a TokenManager class with `get_token()` method that handles caching and refresh.",
            "Use `httpx.AsyncClient` for async support.",
        ],
        ground_truth=(
            "TokenManager class caches token, refreshes 60s before expiry. "
            "POST to token endpoint with client_credentials grant type. "
            "Store expiry as `time.time() + expires_in`."
        ),
        skill_keywords=["OAuth 2.0", "client credentials", "access token", "refresh", "TokenManager", "expiry"],
        tool_names=["code_edit"],
    ),
    Task(
        id="api_integration_02",
        domain="api_integration",
        difficulty="medium",
        title="Handle API pagination with cursor-based navigation",
        user_query=(
            "The API I'm calling returns paginated results with a `next_cursor` field in the response. "
            "When `next_cursor` is null, there are no more pages. "
            "How do I fetch all pages and collect all results?"
        ),
        solution_steps=[
            "Initialize: `cursor = None`, `all_items = []`.",
            "Loop: while True → call API with `cursor` param → extend all_items → "
            "break if `response['next_cursor'] is None` else `cursor = response['next_cursor']`.",
            "Add rate limiting: `await asyncio.sleep(0.1)` between pages.",
            "Add max_pages guard to prevent infinite loops.",
        ],
        ground_truth=(
            "While-loop with cursor param; break when next_cursor is None. "
            "Add sleep between pages and a max_pages safeguard."
        ),
        skill_keywords=["pagination", "cursor", "next_cursor", "while loop", "rate limit", "collect all"],
        tool_names=["code_edit"],
    ),
    Task(
        id="api_integration_03",
        domain="api_integration",
        difficulty="medium",
        title="Implement exponential backoff retry",
        user_query=(
            "My API calls sometimes fail with 429 (rate limited) or 503 (service unavailable). "
            "How do I implement retry with exponential backoff and jitter?"
        ),
        solution_steps=[
            "Retry on 429 and 503 only; raise immediately on 4xx client errors.",
            "Backoff: wait = `min(base * 2**attempt + random.uniform(0, 1), max_wait)`.",
            "Show a decorator or a `retry_request()` helper.",
            "Read `Retry-After` header on 429 if present.",
            "Recommend: use `tenacity` library for production retry logic.",
        ],
        ground_truth=(
            "Retry on 429/503 with exponential backoff: `base * 2**attempt + jitter`. "
            "Respect Retry-After header. Max attempts = 5, max_wait = 60s. Use tenacity in production."
        ),
        skill_keywords=["retry", "exponential backoff", "jitter", "429", "503", "Retry-After", "tenacity"],
        tool_names=["code_edit"],
    ),
    Task(
        id="api_integration_04",
        domain="api_integration",
        difficulty="easy",
        title="Validate and parse a JSON API response",
        user_query=(
            "I'm getting a JSON response from an API and I want to validate its structure "
            "before using it. The required fields are: `id` (int), `email` (string), `active` (bool). "
            "How do I validate this in Python?"
        ),
        solution_steps=[
            "Option 1: pydantic model — `class User(BaseModel): id: int; email: str; active: bool`. "
            "Parse: `user = User(**response_json)` → raises ValidationError on bad input.",
            "Option 2: jsonschema library with a schema dict.",
            "Recommend pydantic for Python-native use; jsonschema for language-agnostic schemas.",
            "Always wrap parse in try/except and return a structured error.",
        ],
        ground_truth=(
            "Use pydantic BaseModel: `class User(BaseModel): id: int; email: str; active: bool`. "
            "Instantiate with `User(**json_data)`. Catches type errors and missing fields."
        ),
        skill_keywords=["pydantic", "validation", "BaseModel", "ValidationError", "JSON schema", "type checking"],
        tool_names=["code_edit"],
    ),
    Task(
        id="api_integration_05",
        domain="api_integration",
        difficulty="medium",
        title="Set up a webhook receiver in FastAPI",
        user_query=(
            "I need to receive webhooks from a payment provider. "
            "The provider sends POST requests with a JSON body and a signature header "
            "`X-Signature: sha256=<hmac>`. How do I verify the signature and process events?"
        ),
        solution_steps=[
            "FastAPI endpoint: `@app.post('/webhook')` receiving `Request` directly (not parsed body) "
            "to get raw bytes for signature verification.",
            "Verify: `hmac.compare_digest(expected_sig, received_sig)` using the raw body bytes and secret.",
            "Compute expected: `hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()`.",
            "Return 200 quickly; process event in a background task.",
            "Return 400 on invalid signature to prevent processing forged events.",
        ],
        ground_truth=(
            "Read raw bytes before parsing JSON; compute HMAC-SHA256 with shared secret; "
            "compare_digest to prevent timing attacks; return 200 immediately and process async."
        ),
        skill_keywords=["webhook", "HMAC", "signature", "compare_digest", "timing attack", "raw body", "FastAPI"],
        tool_names=["code_edit"],
    ),
    Task(
        id="api_integration_06",
        domain="api_integration",
        difficulty="medium",
        title="Handle rate limit headers and respect limits",
        user_query=(
            "The API I call returns `X-RateLimit-Remaining` and `X-RateLimit-Reset` headers. "
            "How do I use these to avoid hitting the rate limit?"
        ),
        solution_steps=[
            "After each response, read headers: `remaining = int(response.headers.get('X-RateLimit-Remaining', 100))`.",
            "Read reset time: `reset_at = int(response.headers.get('X-RateLimit-Reset', 0))`.",
            "If remaining < threshold (e.g., 5): sleep until reset_at.",
            "Sleep: `await asyncio.sleep(max(0, reset_at - time.time()) + 1)`.",
            "Wrap in an `APIClient` class that tracks these automatically.",
        ],
        ground_truth=(
            "Read X-RateLimit-Remaining after each call; if < 5, sleep until X-RateLimit-Reset epoch. "
            "Add 1s buffer after reset time."
        ),
        skill_keywords=["rate limit", "X-RateLimit-Remaining", "X-RateLimit-Reset", "sleep", "headers", "epoch"],
        tool_names=["code_edit"],
    ),
    Task(
        id="api_integration_07",
        domain="api_integration",
        difficulty="easy",
        title="Authenticate with API key in header",
        user_query=(
            "I need to call an API that requires an API key in the `Authorization: Bearer <key>` header. "
            "How do I set this up with httpx for all requests?"
        ),
        solution_steps=[
            "Use `httpx.AsyncClient` with `headers` parameter.",
            "Show: `async with httpx.AsyncClient(headers={'Authorization': f'Bearer {api_key}'}) as client:`.",
            "Store api_key in environment variable, not in code.",
            "Use `python-dotenv` to load from `.env` file in development.",
        ],
        ground_truth=(
            "Pass `headers={'Authorization': f'Bearer {api_key}'}` to httpx.AsyncClient constructor. "
            "Load api_key from environment variable, never hardcode."
        ),
        skill_keywords=["Bearer token", "Authorization header", "httpx", "environment variable", "API key"],
        tool_names=["code_edit"],
    ),
    Task(
        id="api_integration_08",
        domain="api_integration",
        difficulty="medium",
        title="Upload a file via multipart form POST",
        user_query=(
            "I need to upload a PDF file to an API endpoint that expects multipart/form-data. "
            "The field name is `document`, and I also need to send a `description` text field. "
            "How do I do this with httpx?"
        ),
        solution_steps=[
            "Open file in binary mode: `with open('report.pdf', 'rb') as f:`.",
            "Use httpx files param: `files = {'document': ('report.pdf', f, 'application/pdf')}`.",
            "Add data param for text fields: `data = {'description': 'Q4 Report'}`.",
            "POST: `response = await client.post(url, files=files, data=data)`.",
            "Do not set Content-Type manually — httpx sets it automatically with boundary.",
        ],
        ground_truth=(
            "Pass `files={'document': (filename, file_obj, 'application/pdf')}` and `data={'description': '..'}` "
            "to httpx.post(). Do not set Content-Type header manually."
        ),
        skill_keywords=["multipart", "file upload", "httpx", "files param", "Content-Type", "boundary"],
        tool_names=["code_edit"],
    ),
    Task(
        id="api_integration_09",
        domain="api_integration",
        difficulty="hard",
        title="Stream a large API response efficiently",
        user_query=(
            "I need to call an API that returns a very large JSON-Lines response (potentially GB). "
            "I cannot load it all into memory. How do I stream and process it line by line?"
        ),
        solution_steps=[
            "Use httpx streaming: `async with client.stream('GET', url) as response:`.",
            "Iterate lines: `async for line in response.aiter_lines():`.",
            "Parse each line as JSON: `record = json.loads(line)`.",
            "Process record immediately; do not accumulate.",
            "Handle empty lines (some APIs emit keepalive newlines): `if not line.strip(): continue`.",
        ],
        ground_truth=(
            "Use `httpx.stream()` context manager + `aiter_lines()` to process one line at a time. "
            "Skip empty lines. Parse each line as JSON immediately."
        ),
        skill_keywords=["streaming", "httpx", "aiter_lines", "JSON Lines", "memory efficient", "large response"],
        tool_names=["code_edit"],
    ),
    Task(
        id="api_integration_10",
        domain="api_integration",
        difficulty="medium",
        title="Handle API versioning in client code",
        user_query=(
            "The API I depend on is releasing v2, which breaks backward compatibility. "
            "I need to support both v1 and v2 in my client during a transition period. "
            "How should I structure this?"
        ),
        solution_steps=[
            "Create an abstract base client with the shared interface.",
            "Create `APIClientV1` and `APIClientV2` subclasses.",
            "Factory function: `get_client(version='v1')` returns the appropriate subclass.",
            "Route-level versioning: prefix URLs with `/v1/` or `/v2/` in each subclass.",
            "Feature flag: read version from config; allow per-endpoint override.",
        ],
        ground_truth=(
            "Abstract base client with V1/V2 subclasses. Factory returns correct version from config. "
            "URL prefix versioning: `/api/v1/` vs `/api/v2/`."
        ),
        skill_keywords=["API versioning", "abstract base", "factory", "URL prefix", "backward compatibility"],
        tool_names=["code_edit"],
    ),
]

# ── Domain: sys_admin ─────────────────────────────────────────────────────────

_SYS_ADMIN: List[Task] = [
    Task(
        id="sys_admin_01",
        domain="sys_admin",
        difficulty="medium",
        title="Diagnose high CPU usage on Linux",
        user_query="The server CPU is at 95% and I need to find which process is causing it. Walk me through the diagnosis.",
        solution_steps=[
            "Run `top` or `htop` to identify the highest-CPU PID.",
            "Get process details: `ps aux --sort=-%cpu | head -20`.",
            "For a multi-threaded process: `top -H -p <PID>` to see per-thread CPU.",
            "Check if it's a legitimate spike: `uptime` to see load average trend.",
            "If it's a Python process: attach `py-spy top --pid <PID>` to see the hot call stack.",
        ],
        ground_truth=(
            "Use top/htop to find PID, ps aux for details, top -H for threads. "
            "Use py-spy for Python processes to get live call stack."
        ),
        skill_keywords=["top", "htop", "ps aux", "CPU", "PID", "py-spy", "load average"],
        tool_names=["bash"],
    ),
    Task(
        id="sys_admin_02",
        domain="sys_admin",
        difficulty="easy",
        title="Find and free disk space",
        user_query="My server disk is at 98%. How do I find what is using the space and clean it up safely?",
        solution_steps=[
            "Check overall usage: `df -h` to see which filesystem is full.",
            "Find large directories: `du -sh /* | sort -hr | head -20`.",
            "Find large files: `find /var -type f -size +500M`.",
            "Common culprits: Docker images (`docker system prune`), log files (`/var/log`), package cache.",
            "Rotate logs: `journalctl --vacuum-size=500M`.",
        ],
        ground_truth=(
            "df -h → du -sh to find culprit directory → find large files → "
            "docker system prune / journalctl --vacuum-size for common culprits."
        ),
        skill_keywords=["df", "du", "disk space", "docker prune", "journalctl", "vacuum", "large files"],
        tool_names=["bash"],
    ),
    Task(
        id="sys_admin_03",
        domain="sys_admin",
        difficulty="medium",
        title="Configure nginx as a reverse proxy",
        user_query=(
            "I have a Python app running on `127.0.0.1:8000`. "
            "Set up nginx to proxy `example.com` to it with HTTPS (cert already exists at `/etc/ssl/`)."
        ),
        solution_steps=[
            "Create `/etc/nginx/sites-available/example.com` with upstream block.",
            "HTTP block: redirect 80 → 443.",
            "HTTPS block: ssl_certificate, ssl_certificate_key, proxy_pass to 127.0.0.1:8000.",
            "Set proxy headers: `proxy_set_header Host $host`, `X-Real-IP $remote_addr`, `X-Forwarded-For`.",
            "Enable site: `ln -s /etc/nginx/sites-available/example.com /etc/nginx/sites-enabled/` → nginx -t → systemctl reload nginx.",
        ],
        ground_truth=(
            "nginx config with HTTP→HTTPS redirect, ssl_certificate/key paths, proxy_pass to 127.0.0.1:8000, "
            "proxy headers set. Enable via symlink + nginx -t + reload."
        ),
        skill_keywords=["nginx", "reverse proxy", "proxy_pass", "HTTPS", "ssl_certificate", "proxy_set_header"],
        tool_names=["bash", "file_write"],
    ),
    Task(
        id="sys_admin_04",
        domain="sys_admin",
        difficulty="easy",
        title="Schedule a cron job",
        user_query=(
            "I want to run `/opt/scripts/cleanup.sh` every day at 2:30 AM as the `deploy` user. "
            "How do I set up the cron job and make sure the output is logged?"
        ),
        solution_steps=[
            "Edit crontab for deploy user: `crontab -u deploy -e`.",
            "Add line: `30 2 * * * /opt/scripts/cleanup.sh >> /var/log/cleanup.log 2>&1`.",
            "Verify syntax with `crontab -l -u deploy`.",
            "Make sure the script is executable: `chmod +x /opt/scripts/cleanup.sh`.",
            "Note: cron uses minimal environment — specify full paths in the script.",
        ],
        ground_truth=(
            "crontab line: `30 2 * * * /opt/scripts/cleanup.sh >> /var/log/cleanup.log 2>&1`. "
            "Run as deploy user. Script must be executable. Use full paths inside script."
        ),
        skill_keywords=["cron", "crontab", "schedule", "redirect stderr", "2>&1", "executable", "deploy user"],
        tool_names=["bash"],
    ),
    Task(
        id="sys_admin_05",
        domain="sys_admin",
        difficulty="medium",
        title="Debug SSH connection timeout",
        user_query=(
            "SSH to a remote server hangs for 30 seconds then connects, or sometimes fails entirely. "
            "How do I diagnose and fix this?"
        ),
        solution_steps=[
            "Run `ssh -vvv user@host` to see verbose negotiation; identify where it stalls.",
            "Common cause 1: DNS reverse lookup. Fix: `UseDNS no` in `/etc/ssh/sshd_config`.",
            "Common cause 2: GSSAPI auth attempt. Fix: `GSSAPIAuthentication no` in sshd_config.",
            "Common cause 3: firewall dropping packets (no RST, so TCP times out) — check route MTU.",
            "Reload sshd after config change: `systemctl reload sshd`.",
        ],
        ground_truth=(
            "ssh -vvv to identify stall point. Most common: UseDNS=yes (add UseDNS no) or "
            "GSSAPIAuthentication (add GSSAPIAuthentication no) in sshd_config. Reload sshd."
        ),
        skill_keywords=["SSH", "timeout", "UseDNS", "GSSAPIAuthentication", "sshd_config", "verbose", "dns lookup"],
        tool_names=["bash"],
    ),
    Task(
        id="sys_admin_06",
        domain="sys_admin",
        difficulty="medium",
        title="Configure ufw firewall rules",
        user_query=(
            "Set up ufw on an Ubuntu server to: allow SSH (port 22), HTTP (80), HTTPS (443), "
            "allow port 5432 only from IP 10.0.1.5, and deny everything else by default."
        ),
        solution_steps=[
            "`ufw default deny incoming`.",
            "`ufw allow 22/tcp` (SSH).",
            "`ufw allow 80/tcp` (HTTP).",
            "`ufw allow 443/tcp` (HTTPS).",
            "`ufw allow from 10.0.1.5 to any port 5432 proto tcp` (PostgreSQL from trusted IP).",
            "`ufw enable` → `ufw status verbose` to confirm.",
        ],
        ground_truth=(
            "ufw default deny → allow 22, 80, 443 → allow from 10.0.1.5 port 5432 → ufw enable."
        ),
        skill_keywords=["ufw", "firewall", "deny incoming", "allow from", "port 5432", "PostgreSQL", "whitelist"],
        tool_names=["bash"],
    ),
    Task(
        id="sys_admin_07",
        domain="sys_admin",
        difficulty="easy",
        title="Monitor and search log files in real time",
        user_query="How do I watch a log file in real time and filter lines containing the word 'ERROR'?",
        solution_steps=[
            "Basic tail: `tail -f /var/log/app.log`.",
            "With grep filter: `tail -f /var/log/app.log | grep --line-buffered 'ERROR'`.",
            "Case-insensitive: `grep -i`.",
            "Highlight matches: `grep --color=always`.",
            "For structured JSON logs: `tail -f app.log | jq 'select(.level == \"ERROR\")'`.",
        ],
        ground_truth=(
            "Use `tail -f logfile | grep --line-buffered 'ERROR'`. "
            "For JSON logs use `tail -f | jq 'select(.level==\"ERROR\")'`."
        ),
        skill_keywords=["tail -f", "grep", "line-buffered", "real time", "log monitoring", "jq", "JSON logs"],
        tool_names=["bash"],
    ),
    Task(
        id="sys_admin_08",
        domain="sys_admin",
        difficulty="medium",
        title="Create a systemd service for a Python app",
        user_query=(
            "How do I create a systemd service unit to run `/opt/app/server.py` "
            "as the `appuser` user, restart on failure, and start at boot?"
        ),
        solution_steps=[
            "Create `/etc/systemd/system/myapp.service` with [Unit], [Service], [Install] sections.",
            "[Service]: ExecStart=/usr/bin/python3 /opt/app/server.py, User=appuser, Restart=on-failure, RestartSec=5.",
            "[Install]: WantedBy=multi-user.target.",
            "Reload and enable: `systemctl daemon-reload && systemctl enable --now myapp`.",
            "Check status: `systemctl status myapp` and `journalctl -u myapp -f`.",
        ],
        ground_truth=(
            "systemd unit with User=appuser, Restart=on-failure, WantedBy=multi-user.target. "
            "daemon-reload → enable --now → check with journalctl -u myapp."
        ),
        skill_keywords=["systemd", "service unit", "Restart=on-failure", "WantedBy", "daemon-reload", "journalctl"],
        tool_names=["bash", "file_write"],
    ),
    Task(
        id="sys_admin_09",
        domain="sys_admin",
        difficulty="easy",
        title="Debug 'Permission denied' on a file",
        user_query=(
            "My app gets 'Permission denied' trying to write to `/var/data/output.json`. "
            "The app runs as user `appuser`. How do I diagnose and fix this?"
        ),
        solution_steps=[
            "Check current permissions: `ls -la /var/data/`.",
            "Check file owner and group: `stat /var/data/output.json`.",
            "Check if appuser has write access: `sudo -u appuser test -w /var/data/output.json && echo ok`.",
            "Fix option 1: `chown appuser /var/data/output.json`.",
            "Fix option 2: `chmod 664 /var/data/output.json` and add appuser to the owning group.",
            "For directories: also need execute (x) permission to traverse.",
        ],
        ground_truth=(
            "ls -la to check permissions, stat for owner. Fix: chown appuser or add user to owning group + chmod. "
            "Directories need x permission to traverse."
        ),
        skill_keywords=["permission denied", "ls -la", "chown", "chmod", "stat", "group", "execute bit"],
        tool_names=["bash"],
    ),
    Task(
        id="sys_admin_10",
        domain="sys_admin",
        difficulty="easy",
        title="Set and persist environment variables for a service",
        user_query=(
            "How do I set environment variables (`DATABASE_URL`, `SECRET_KEY`) "
            "for a systemd service so they persist across reboots and are not visible in `ps aux`?"
        ),
        solution_steps=[
            "Create an environment file: `/etc/myapp/env` with `DATABASE_URL=...` and `SECRET_KEY=...` on separate lines.",
            "Set permissions: `chmod 600 /etc/myapp/env` (only root can read).",
            "Reference in service unit: `EnvironmentFile=/etc/myapp/env` in [Service] section.",
            "Systemctl daemon-reload → restart service.",
            "Verify: `systemctl show myapp | grep Environment` (values will be shown — use systemd credentials for secrets in production).",
        ],
        ground_truth=(
            "EnvironmentFile=/etc/myapp/env in [Service]. File chmod 600. "
            "Variables are NOT shown in ps aux but ARE visible to root via systemctl show."
        ),
        skill_keywords=["EnvironmentFile", "systemd", "environment variables", "chmod 600", "secrets", "ps aux"],
        tool_names=["bash", "file_write"],
    ),
]

# ── Domain: qa ────────────────────────────────────────────────────────────────

_QA: List[Task] = [
    Task(
        id="qa_01",
        domain="qa",
        difficulty="easy",
        title="Explain async/await in Python",
        user_query="Can you explain how async/await works in Python and when I should use it?",
        solution_steps=[
            "Explain the event loop: asyncio runs a single-threaded event loop that switches between coroutines at await points.",
            "Explain coroutines: `async def` functions that can be suspended with `await`.",
            "Use async when I/O is the bottleneck (network, disk), not CPU. For CPU-bound work use multiprocessing.",
            "Show a simple example: `async def fetch(url): async with httpx.AsyncClient() as c: return await c.get(url)`.",
            "Explain common mistake: blocking I/O inside async function blocks the entire event loop.",
        ],
        ground_truth=(
            "async/await uses an event loop; coroutines suspend at await points. "
            "Use for I/O-bound work, not CPU-bound. Never call blocking functions inside async functions."
        ),
        skill_keywords=["async", "await", "event loop", "coroutine", "I/O-bound", "CPU-bound", "blocking"],
        tool_names=[],
    ),
    Task(
        id="qa_02",
        domain="qa",
        difficulty="medium",
        title="Compare Redis vs Memcached for session caching",
        user_query="When should I choose Redis over Memcached for caching user sessions? What are the key differences?",
        solution_steps=[
            "Data structures: Redis supports strings, hashes, lists, sets, sorted sets. Memcached: strings only.",
            "Persistence: Redis supports RDB/AOF persistence. Memcached: in-memory only.",
            "Replication: Redis has built-in replication and Sentinel/Cluster. Memcached: no native replication.",
            "For sessions: Redis is preferred — richer data model (hash per session), persistence option, pub/sub for invalidation.",
            "Memcached advantage: simpler, slightly lower latency at very high throughput.",
        ],
        ground_truth=(
            "Choose Redis for sessions: richer data types, optional persistence, built-in replication. "
            "Memcached is simpler and faster but no persistence, no replication, strings only."
        ),
        skill_keywords=["Redis", "Memcached", "session", "persistence", "replication", "data structures", "pub/sub"],
        tool_names=[],
    ),
    Task(
        id="qa_03",
        domain="qa",
        difficulty="medium",
        title="Explain the Observer design pattern",
        user_query="Can you explain the Observer pattern and give a Python example of when to use it?",
        solution_steps=[
            "Definition: Observer defines a one-to-many dependency so that when one object changes state, "
            "all dependents are notified automatically.",
            "Components: Subject (publisher) maintains a list of Observers (subscribers); "
            "calls notify() on all of them when state changes.",
            "Python example: event system for a domain model — `Order` notifies `InventoryService` and `EmailService` on status change.",
            "Show a simple implementation with register/unregister/notify methods.",
            "Note: Python's `asyncio.Event` or `threading.Event` are built-in observer-like primitives.",
        ],
        ground_truth=(
            "Observer: Subject notifies all registered Observers on state change. "
            "Python example: Order.notify() calls inventory and email handlers. "
            "asyncio.Event is a built-in primitive."
        ),
        skill_keywords=["Observer", "publisher", "subscriber", "notify", "event", "design pattern", "asyncio.Event"],
        tool_names=[],
    ),
    Task(
        id="qa_04",
        domain="qa",
        difficulty="easy",
        title="Describe Git trunk-based development best practices",
        user_query="What is trunk-based development and how is it different from Gitflow?",
        solution_steps=[
            "Trunk-based: all developers commit directly to main (trunk) or to very short-lived feature branches (< 2 days).",
            "Gitflow: long-lived feature branches (days/weeks), release branches, hotfix branches.",
            "Trunk-based advantages: fewer merge conflicts, faster CI feedback, enables continuous delivery.",
            "Requirements for trunk-based: feature flags for incomplete features, strong CI pipeline, pair programming.",
            "Gitflow advantages: good for versioned releases (libraries, mobile apps) where you support multiple versions.",
        ],
        ground_truth=(
            "Trunk-based: short-lived branches (<2 days), merge to main frequently. "
            "Requires feature flags + strong CI. Gitflow better for versioned products. "
            "Trunk-based enables continuous delivery."
        ),
        skill_keywords=["trunk-based", "Gitflow", "feature flags", "continuous delivery", "merge conflicts", "CI"],
        tool_names=[],
    ),
    Task(
        id="qa_05",
        domain="qa",
        difficulty="medium",
        title="Explain HTTP/2 vs HTTP/1.1",
        user_query="What are the main improvements in HTTP/2 over HTTP/1.1 and when does it matter?",
        solution_steps=[
            "Multiplexing: HTTP/2 sends multiple requests over one TCP connection simultaneously. "
            "HTTP/1.1 needs multiple connections or queues requests.",
            "Header compression: HTTP/2 uses HPACK to compress headers. HTTP/1.1 sends headers as plain text each time.",
            "Server push: HTTP/2 allows server to proactively send resources.",
            "Binary framing: HTTP/2 uses binary protocol (more efficient to parse).",
            "When it matters: many small requests (API calls, assets). Less benefit for large single-file transfers.",
        ],
        ground_truth=(
            "HTTP/2 advantages: multiplexing (one TCP connection), HPACK header compression, binary framing, server push. "
            "Most impactful for many concurrent small requests."
        ),
        skill_keywords=["HTTP/2", "multiplexing", "HPACK", "server push", "binary framing", "TCP connection"],
        tool_names=[],
    ),
    Task(
        id="qa_06",
        domain="qa",
        difficulty="medium",
        title="JWT vs server-side sessions for authentication",
        user_query="Should I use JWTs or server-side sessions for user authentication in a web API?",
        solution_steps=[
            "JWT: stateless — server verifies signature without DB lookup. Scales horizontally. "
            "Downside: cannot revoke before expiry; token grows with claims.",
            "Server-side sessions: stored in DB/Redis; easily revocable. Requires session store. "
            "Scales with Redis but adds infrastructure.",
            "JWT best for: microservices, mobile clients, multi-service token passing.",
            "Sessions best for: traditional web apps where you need instant revocation (logout, account ban).",
            "Common JWT mistake: long expiry without refresh token rotation.",
        ],
        ground_truth=(
            "JWT: stateless, scalable, not revocable. Sessions: revocable, requires store. "
            "Use JWT for microservices; sessions for apps needing instant revocation. "
            "Always use short JWT expiry + refresh token rotation."
        ),
        skill_keywords=["JWT", "session", "stateless", "revocation", "refresh token", "expiry", "microservices"],
        tool_names=[],
    ),
    Task(
        id="qa_07",
        domain="qa",
        difficulty="hard",
        title="Explain B-tree index internals",
        user_query="How does a B-tree database index work internally, and why does column order matter in composite indexes?",
        solution_steps=[
            "B-tree: balanced tree where each node holds sorted keys and pointers to children. "
            "Leaf nodes hold the actual data pointers. Height is O(log n).",
            "Database lookup: traverse from root → compare key → follow pointer → leaf → fetch row.",
            "Range scans: leaf nodes are linked; range query follows leaf chain.",
            "Composite index (a, b, c): sorted first by a, then b within same a, then c. "
            "A WHERE clause on `b` alone cannot use the index (no leading column). "
            "A WHERE on `a` alone can — it uses the first column of the sorted order.",
            "Rule: put the most selective column first, OR the column that appears in WHERE clause first.",
        ],
        ground_truth=(
            "B-tree is a balanced sorted tree; O(log n) lookup. Leaf nodes are linked for range scans. "
            "Composite index (a,b,c): query must include a leading prefix. WHERE on b alone skips the index."
        ),
        skill_keywords=["B-tree", "composite index", "leading prefix", "range scan", "leaf node", "selectivity"],
        tool_names=[],
    ),
    Task(
        id="qa_08",
        domain="qa",
        difficulty="medium",
        title="TDD vs BDD — when to use each",
        user_query="What is the difference between TDD and BDD, and when should I use each approach?",
        solution_steps=[
            "TDD (Test-Driven Development): write a failing unit test → write minimal code to pass → refactor. "
            "Cycle: Red → Green → Refactor.",
            "BDD (Behavior-Driven Development): write tests in business language (Given/When/Then). "
            "Tools: pytest-bdd, behave in Python.",
            "TDD: developer-focused; tests are code-level; faster inner loop.",
            "BDD: stakeholder-focused; tests document business requirements; bridges dev and product.",
            "Use TDD for: internal code, algorithms, APIs. BDD for: acceptance testing, features with product owner.",
        ],
        ground_truth=(
            "TDD: Red-Green-Refactor, unit-level, developer-focused. "
            "BDD: Given/When/Then, business language, bridges dev and product. "
            "Use TDD for internal logic; BDD for acceptance criteria."
        ),
        skill_keywords=["TDD", "BDD", "Red-Green-Refactor", "Given/When/Then", "acceptance test", "behave"],
        tool_names=[],
    ),
    Task(
        id="qa_09",
        domain="qa",
        difficulty="medium",
        title="Explain blue-green deployment",
        user_query="What is a blue-green deployment and what are its advantages and limitations?",
        solution_steps=[
            "Blue-green: maintain two identical production environments (blue = current, green = new version).",
            "Deploy to green; run smoke tests on green while blue serves all traffic.",
            "Switch traffic to green (load balancer change, DNS update, or feature flag).",
            "Blue stays as instant rollback — switch back if green has issues.",
            "Advantages: zero-downtime deployment, instant rollback.",
            "Limitations: double infrastructure cost, database schema changes require care (both versions share DB during cutover).",
        ],
        ground_truth=(
            "Two identical environments; switch traffic at load balancer. Instant rollback by switching back. "
            "Limitation: DB schema changes must be backward compatible during cutover; double infra cost."
        ),
        skill_keywords=["blue-green", "zero-downtime", "rollback", "load balancer", "smoke test", "DB schema"],
        tool_names=[],
    ),
    Task(
        id="qa_10",
        domain="qa",
        difficulty="hard",
        title="Actors vs CSP concurrency models",
        user_query=(
            "What is the difference between the Actor model and CSP (Communicating Sequential Processes)? "
            "When would you choose one over the other?"
        ),
        solution_steps=[
            "Actor model: entities (actors) communicate by sending messages to each other's mailboxes. "
            "No shared memory. Each actor processes one message at a time. Examples: Erlang/Elixir, Akka.",
            "CSP: concurrent processes communicate through channels. Channels can be buffered or synchronous. "
            "Examples: Go goroutines + channels, Clojure core.async.",
            "Key difference: Actors address messages to a specific actor (identity-based). "
            "CSP addresses messages to a channel (channel-based; sender does not know receiver).",
            "Actor model better for: distributed systems, fault tolerance (Erlang supervisors), "
            "dynamic topologies.",
            "CSP better for: local concurrency, pipeline patterns (data flowing through stages), "
            "Go-style services.",
        ],
        ground_truth=(
            "Actors: message to actor mailbox (identity-based), no shared memory, built-in fault tolerance. "
            "CSP: message through channel (channel-based), synchronous or buffered. "
            "Actors for distributed; CSP for local pipelines."
        ),
        skill_keywords=["Actor model", "CSP", "channel", "mailbox", "Go", "Erlang", "goroutine", "distributed"],
        tool_names=[],
    ),
]

# ── Assemble ──────────────────────────────────────────────────────────────────

_ALL_TASKS: List[Task] = (
    _CODE_DEBUG
    + _DOC_DRAFT
    + _DATA_ANALYSIS
    + _API_INTEGRATION
    + _SYS_ADMIN
    + _QA
)

TASK_BANK: TaskBank = TaskBank(_ALL_TASKS)
