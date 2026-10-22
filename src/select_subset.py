"""Build two pilot manifests from Galaxy Zoo 2's real volunteer labels.

Class6.1 / Class6.2 are the actual GZ2 decision-tree vote fractions for
"Is there anything odd?" (Yes / No) — this uses real crowd-sourced labels,
not an invented heuristic.

- vlm_pilot: extreme cases (most-unanimously-odd + most-unanimously-ordinary),
  good for a human-readable qualitative demo of what the chat-style VLMs say.
- lof_pilot: a plain random sample (not pre-filtered), so the embedding-based
  anomaly scores can be correlated against Class6.1 without selection bias.
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "pilot_manifests"


def build(n_vlm_per_class=25, n_lof=300, seed=42):
    df = pd.read_csv(RAW_DIR / "training_solutions_rev1.csv")

    odd = df.sort_values("Class6.1", ascending=False).head(n_vlm_per_class).copy()
    odd["human_label"] = "odd"
    ordinary = df.sort_values("Class6.2", ascending=False).head(n_vlm_per_class).copy()
    ordinary["human_label"] = "ordinary"
    vlm_pilot = pd.concat([odd, ordinary])[["GalaxyID", "Class6.1", "Class6.2", "human_label"]]

    lof_pilot = df.sample(n=n_lof, random_state=seed)[["GalaxyID", "Class6.1", "Class6.2"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vlm_pilot.to_csv(OUT_DIR / "vlm_pilot.csv", index=False)
    lof_pilot.to_csv(OUT_DIR / "lof_pilot.csv", index=False)
    print(f"vlm_pilot: {len(vlm_pilot)} rows -> {OUT_DIR / 'vlm_pilot.csv'}")
    print(f"lof_pilot: {len(lof_pilot)} rows -> {OUT_DIR / 'lof_pilot.csv'}")


if __name__ == "__main__":
    build()
