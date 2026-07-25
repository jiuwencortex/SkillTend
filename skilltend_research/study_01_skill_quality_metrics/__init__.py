# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Study 01 — Skill Quality Metrics Validation.

Validates that the four Q sub-metrics (RR, ID, TSR, PS) are:
  - Valid proxies for real skill utility (correlate with human labels, Spearman ρ > 0.5)
  - Stable across repeated measurements (CV < 0.08 for RR/ID/PS, < 0.15 for TSR)
  - Not redundant with each other (pairwise r < 0.92)

Entry points:
  runner.py            — orchestrates the full validation pipeline
  label_tool.py        — interactive CLI for human labelling of 30 skills
  compute_validity.py  — computes metric correlations after labelling
  calibrate_weights.py — fits Q weights using the labelled dataset
"""
