"""Real embedding-based anomaly detection: extract image embeddings from a
model, score each image with Local Outlier Factor (a standard, established
novelty-detection method), and check whether that score tracks the real
Galaxy Zoo 2 volunteer "is there anything odd?" vote fraction (Class6.1).

Writes one row per image per model into the same fixed results schema
(image_id, model_name, prompt_id, answer, timestamp), and a short markdown
report with the correlation finding per model.
"""
import argparse
import csv
import gc
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.neighbors import LocalOutlierFactor

from data import DEFAULT_DATA_DIR, get_manifest_subset
from models import EMBEDDING_REGISTRY

FIELDNAMES = ["image_id", "model_name", "prompt_id", "answer", "timestamp"]
PROMPT_ID = "lof_v1"

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def run_one_model(model_name, subset, n_neighbors, results_writer, results_file, log_file):
    def log(line):
        log_file.write(line + "\n")
        log_file.flush()

    load_fn, embed_fn = EMBEDDING_REGISTRY[model_name]
    log(f"[{datetime.now(timezone.utc).isoformat()}] loading {model_name}")
    loaded = load_fn()

    embeddings = []
    image_ids = []
    t0 = time.monotonic()
    n = len(subset)
    checkpoint = max(1, n // 20)  # ~20 progress/throughput log lines per model
    for i, (_, row) in enumerate(subset.iterrows(), start=1):
        try:
            image = Image.open(row["image_path"]).convert("RGB")
            embeddings.append(embed_fn(loaded, image))
            image_ids.append(row["GalaxyID"])
        except Exception as exc:
            log(f"  {model_name} failed on {row['GalaxyID']}: {exc!r}")
        if i % checkpoint == 0 or i == n:
            elapsed = time.monotonic() - t0
            log(f"  {model_name}: {i}/{n} images, {elapsed:.1f}s elapsed, {i / elapsed:.2f} img/s")

    del loaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    X = np.stack(embeddings)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    lof.fit_predict(X)
    # negative_outlier_factor_ is <= -1, more negative = more anomalous.
    # Flip sign so higher = more anomalous, the more intuitive convention.
    scores = -lof.negative_outlier_factor_

    now = datetime.now(timezone.utc).isoformat()
    for image_id, score in zip(image_ids, scores):
        results_writer.writerow(
            {
                "image_id": image_id,
                "model_name": model_name,
                "prompt_id": PROMPT_ID,
                "answer": f"lof_anomaly_score={score:.4f}",
                "timestamp": now,
            }
        )
    results_file.flush()
    log(f"[{datetime.now(timezone.utc).isoformat()}] finished {model_name}: {len(image_ids)} embeddings")

    return pd.DataFrame({"GalaxyID": image_ids, f"{model_name}_lof_score": scores})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="zoobot,clip")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--n-neighbors", type=int, default=20)
    args = parser.parse_args()

    models_to_run = [m.strip() for m in args.models.split(",")]
    for m in models_to_run:
        if m not in EMBEDDING_REGISTRY:
            raise ValueError(f"Unknown embedding model {m!r}. Options: {list(EMBEDDING_REGISTRY)}")

    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    manifest_path = Path(args.manifest)
    subset = get_manifest_subset(manifest_path, data_dir=data_dir)

    RESULTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = RESULTS_DIR / f"results_lof_{run_id}.csv"
    log_path = LOGS_DIR / f"run_lof_{run_id}.log"
    report_path = RESULTS_DIR / f"lof_correlation_{run_id}.md"

    score_frames = []
    with (
        results_path.open("w", newline="", encoding="utf-8") as results_file,
        log_path.open("w", encoding="utf-8") as log_file,
    ):
        writer = csv.DictWriter(results_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        results_file.flush()

        for model_name in models_to_run:
            try:
                score_frames.append(run_one_model(model_name, subset, args.n_neighbors, writer, results_file, log_file))
            except Exception as exc:
                log_file.write(f"  FAILED model {model_name}: {exc!r}\n")
                log_file.flush()

    # Correlate each model's LOF score against the real human "odd" vote
    # fraction (Class6.1) across the same image set.
    lines = [
        "# Embedding-based anomaly score vs. real GZ2 volunteer labels\n",
        f"Sample size: {len(subset)} galaxies, randomly drawn (not pre-filtered).\n",
        f"Method: Local Outlier Factor (n_neighbors={args.n_neighbors}) on raw model embeddings.\n",
        "Higher LOF score = more of an outlier in that model's embedding space.\n",
        "Compared against Class6.1 = the real GZ2 volunteer vote fraction for \"is there anything odd?\".\n",
        "\n| model | Spearman r (LOF score vs Class6.1) | p-value |",
        "|---|---|---|",
    ]
    merged = subset[["GalaxyID", "Class6.1"]].copy()
    for frame in score_frames:
        merged = merged.merge(frame, on="GalaxyID", how="left")
        score_col = [c for c in frame.columns if c != "GalaxyID"][0]
        valid = merged.dropna(subset=[score_col])
        r, p = spearmanr(valid[score_col], valid["Class6.1"])
        model_name = score_col.replace("_lof_score", "")
        lines.append(f"| {model_name} | {r:.3f} | {p:.4f} |")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote LOF results to {results_path}")
    print(f"Log: {log_path}")
    print(f"Correlation report: {report_path}")


if __name__ == "__main__":
    main()
