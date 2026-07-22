# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Background review prompt texts.

Faithful translations of Hermes's three review prompts into the Jiuwen context.
"""


MEMORY_REVIEW_PROMPT = """\
Review the conversation above and consider saving to memory if appropriate.

Focus on:
1. Has the user revealed things about themselves — their persona, desires,
   preferences, or personal details worth remembering?
2. Has the user expressed expectations about how you should behave, their work
   style, or ways they want you to operate?

If something stands out, save it using the memory_write tool.
If nothing is worth saving, just say 'Nothing to save.' and stop.
"""