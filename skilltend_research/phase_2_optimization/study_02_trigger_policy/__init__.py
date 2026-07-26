# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Study 02 — Trigger Policy Optimization.

Finds the optimal skill_nudge_interval (N) and memory_nudge_interval (M) by
sweeping an N×M grid and computing Q score vs. token cost for each config.

Entry points:
  runner.py           — orchestrates the full sweep + adaptive trigger comparison
  sweep.py            — N×M grid sweep implementation
  adaptive_trigger.py — conversation-distance-based adaptive trigger
  analyze.py          — Pareto frontier plotting and recommendation
"""
