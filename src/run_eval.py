"""Run a batch of (image, model, prompt) VLM predictions and save as CSV.

Examples:
    # Local laptop (no GPU) — run one model at a time to keep memory isolated
    python src/run_eval.py --n-images 5 --models zoobot --data-dir data/raw
    python src/run_eval.py --n-images 5 --models blip2 --data-dir data/raw
    python src/run_eval.py --n-images 5 --models qwen2vl_2b --data-dir data/raw

    # On a Kaggle GPU session
    python src/run_eval.py --n-images 5 --models llava,qwen2vl_7b,blip2,zoobot
"""
import argparse
import csv
import gc
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

from data import DEFAULT_DATA_DIR, get_test_subset
from models import MODEL_REGISTRY

PROMPTS = {
    "p1": (
        "Does this galaxy look like a typical, ordinary galaxy, or does "
        "something about its shape or features look odd or anomalous? "
        "Briefly explain why."
    ),
}

# These models don't take the text prompt at all — they return an
# embedding-based placeholder instead of an answer (see models.py). Tagged
# with their own prompt_id so the CSV makes clear they weren't actually
# asked the p1 question.
EMBEDDING_ONLY_MODELS = {"zoobot", "astroclip"}
EMBEDDING_PROMPT_ID = "embedding_v1"

FIELDNAMES = ["image_id", "model_name", "prompt_id", "answer", "timestamp"]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def run(models_to_run, n_images, prompt_id, data_dir, results_writer, results_file, log_file):
    """Writes each row to results_writer (and flushes) as soon as it's
    produced, and each log line to log_file immediately — so a crash in one
    model (segfault, OOM, etc.) can't erase rows already written for an
    earlier model.
    """
    subset = get_test_subset(n=n_images, data_dir=data_dir)
    prompt_text = PROMPTS[prompt_id]

    def log(line):
        log_file.write(line + "\n")
        log_file.flush()

    n_rows = 0
    for model_name in models_to_run:
        load_fn, predict_fn = MODEL_REGISTRY[model_name]
        log(f"[{datetime.now(timezone.utc).isoformat()}] loading {model_name}")
        try:
            loaded = load_fn()
        except Exception as exc:
            log(f"  FAILED to load {model_name}: {exc!r}")
            continue

        this_prompt_id = EMBEDDING_PROMPT_ID if model_name in EMBEDDING_ONLY_MODELS else prompt_id

        for _, row in subset.iterrows():
            image_id = row["GalaxyID"]
            try:
                image = Image.open(row["image_path"]).convert("RGB")
                answer = predict_fn(loaded, image, prompt_text)
            except Exception as exc:
                answer = f"ERROR: {exc!r}"
                log(f"  {model_name} failed on {image_id}: {exc!r}")

            results_writer.writerow(
                {
                    "image_id": image_id,
                    "model_name": model_name,
                    "prompt_id": this_prompt_id,
                    "answer": answer,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            results_file.flush()
            n_rows += 1

        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f"[{datetime.now(timezone.utc).isoformat()}] finished {model_name}")

    return n_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument(
        "--models",
        type=str,
        default="blip2,zoobot,qwen2vl_2b",
        help=f"Comma-separated subset of: {','.join(MODEL_REGISTRY)}. "
        "On a memory-limited machine, run one at a time rather than combined.",
    )
    parser.add_argument("--prompt-id", type=str, default="p1")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    models_to_run = [m.strip() for m in args.models.split(",")]
    for m in models_to_run:
        if m not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model {m!r}. Options: {list(MODEL_REGISTRY)}")

    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR

    RESULTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = RESULTS_DIR / f"results_{run_id}.csv"
    log_path = LOGS_DIR / f"run_{run_id}.log"

    with (
        results_path.open("w", newline="", encoding="utf-8") as results_file,
        log_path.open("w", encoding="utf-8") as log_file,
    ):
        results_writer = csv.DictWriter(results_file, fieldnames=FIELDNAMES)
        results_writer.writeheader()
        results_file.flush()

        n_rows = run(models_to_run, args.n_images, args.prompt_id, data_dir, results_writer, results_file, log_file)

    print(f"Wrote {n_rows} rows to {results_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
