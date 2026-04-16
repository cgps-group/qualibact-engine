#!/usr/bin/env python
"""Thin shim for backwards compatibility.

The assessment logic lives in qualibact/assess.py. This file exists so that
``python scripts/genome_qc_assess.py`` still works for users who have the old
invocation in their workflows.
"""
import sys
from pathlib import Path

# Ensure the repo root is on sys.path when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qualibact.assess import (  # noqa: E402, F401  (re-exported for importers)
    run_assessment,
    assess_species,
    parse_args,
)

if __name__ == "__main__":
    args = parse_args()
    run_assessment(args)
