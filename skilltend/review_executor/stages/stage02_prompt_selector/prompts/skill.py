# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Background review prompt texts.

Faithful translations of Hermes's three review prompts into the Jiuwen context.
"""


SKILL_REVIEW_PROMPT = """\
Review the conversation and decide whether any skill should be created, updated, \
or patched. You have the full conversation history above.

━━━  PREFERENCE ORDER FOR UPDATES  ━━━
1. UPDATE the skill that was loaded and used during this conversation.
2. UPDATE the umbrella skill whose instructions govern this area of work.
3. ADD a supporting reference file to an existing skill.
4. CREATE a new narrow skill only if no existing skill fits.

━━━  SIGNALS TO LOOK FOR  ━━━
Any one of the following warrants action:

• The user corrected your style, tone, format, legibility, or verbosity.
  Frustration signals such as "stop doing X", "this is too verbose",
  "don't format like this", "just give me the answer", "you always do Y
  and I hate it", or an explicit "remember this" are FIRST-CLASS skill
  signals — act on them immediately.

• The user corrected your workflow, approach, or the sequence of steps you
  took. Encode the correction directly in the skill that governs that work.

• A non-trivial technique, fix, or workaround emerged during this session
  that would help future sessions — capture it as a Troubleshooting or
  Technique note inside the relevant skill.

• A skill that was used during this conversation is wrong, outdated, or
  missing important context — patch it now.

• Two loaded skills cover the same territory — note this for consolidation
  (add a comment in references/, do NOT delete either skill now).

━━━  WHAT NOT TO CAPTURE  ━━━
• Environment-dependent failures (missing binaries, OS-specific setup errors)
• Negative claims about tools ("the browser tool doesn't work here")
• Transient session errors that resolved on retry
• One-off task narratives that will not recur
• Information already present in an existing skill

━━━  PROTECTED SKILLS  ━━━
Do NOT edit skills whose frontmatter contains `immutable: true`.
These are bundled or hub-installed skills.

━━━  OUTPUT FORMAT  ━━━
Call skill_write, skill_patch, or skill_create as many times as needed.
If nothing warrants a change, say 'No skill changes needed.' and stop.
"""