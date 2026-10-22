# Thesis: Can VLMs Detect Anomalous Galaxies in Galaxy Zoo 2?

MSc Data Science thesis (8-month project) investigating whether vision-language
models (VLMs) can reliably detect anomalous galaxies in the Galaxy Zoo 2 dataset.

Models are run on free Kaggle GPUs; results are pulled down and analyzed locally.

**Current phase:** step one — download data, test-run a handful of VLMs, and
compare them for general exposure to the problem. The specific anomaly-detection
research angle (what counts as "anomalous", which models/prompts to focus on)
is still being decided, so later phases of this project may look different.

## Project layout

* `data/` — raw/intermediate Galaxy Zoo 2 data (images, labels). Gitignored.
* `src/` — reusable Python modules.
* `notebooks/` — exploration and analysis notebooks.
* `results/` — model output CSVs.
* `logs/` — run logs.

## Rules

1. **Output format is fixed.** Every model run must be saved as a CSV with
exactly these columns: `image\_id, model\_name, prompt\_id, answer, timestamp`.
2. **Never overwrite existing results.** Always write a new results file
(e.g. include a timestamp or run id in the filename) rather than
overwriting a previous one.
3. **Test on 5 images before scaling up.** Any new or modified script must be
run on a small sample of 5 images first and checked before running on the
full dataset or a full Kaggle GPU session.
4. **Explain code simply.** When asked to explain code, use plain, simple
language — the user is learning and is not an expert coder. Avoid jargon
unless it's defined inline.
5. At the end of every task, update docs/status.md, add any new results to results/summary.md as a short table with a few lines of notes, and commit and push.

