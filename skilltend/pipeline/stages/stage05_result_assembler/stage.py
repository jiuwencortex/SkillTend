from __future__ import annotations

import time
from typing import List, Optional

from ....types import ReviewAction, ReviewResult, ReviewTrigger


def assemble_review_result(
    trigger: ReviewTrigger,
    actions: List[ReviewAction],
    error: Optional[str],
    t0: float,
) -> ReviewResult:
    """Compute elapsed duration, build summary line, and return ReviewResult.

    t0 must be a value previously obtained from time.monotonic().
    """
    duration = time.monotonic() - t0
    summary = " · ".join(a.summary for a in actions) if actions else "No changes"
    return ReviewResult(
        trigger=trigger,
        actions=actions,
        error=error,
        duration_seconds=duration,
        summary_line=summary,
    )
