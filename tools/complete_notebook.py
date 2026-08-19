#!/usr/bin/env python3
"""
Complete the refactored notebook by implementing Cells 7-11.
Reads notebooks/experiment_refactored.ipynb, replaces placeholders,
and writes notebooks/experiment.ipynb.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REFACTORED = REPO / "notebooks" / "experiment_refactored.ipynb"
FINAL = REPO / "notebooks" / "experiment.ipynb"

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

nb = load(REFACTORED)
cells = nb["cells"]

print("Current cells:")
for i, c in enumerate(cells):
    title = c.get("metadata", {}).get("title", "")
    print(f"  {i}: {title} ({sum(len(s) for s in c['source'])} chars)")
