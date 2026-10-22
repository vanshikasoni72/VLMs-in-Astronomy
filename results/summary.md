# Results Summary

| date | model | prompt_id | n_images | results_file | headline | note |
|---|---|---|---|---|---|---|
| 2026-10-22 | zoobot | embedding_v1 | 5 | results_20261022T084947Z.csv | placeholder (embedding norm only) | Encoder-only (timm hub checkpoint); no GZ2 classification head wired up, so this is not a real morphology prediction yet. |
| 2026-10-22 | blip2 | p1 | 5 | results_20261022T084947Z.csv | real text answers | Needed `Question: ... Answer:` framing (it's a continuation model, not chat) plus a repetition penalty to stop it looping/echoing the prompt. |
| 2026-10-22 | qwen2vl_2b | p1 | 5 | results_20261022T084947Z.csv | real text answers | Confidently hallucinates specific object names (e.g. "NGC 1068", "Sgr A*") for anonymous Galaxy Zoo thumbnails — a real reliability finding, not a pipeline bug. |

**Notes**

- First true end-to-end run, on local CPU (no GPU available on this laptop). LLaVA-1.5-7B and Qwen2-VL-7B are still pending — too large to run here, queued for Kaggle (`notebooks/model_smoke_test.ipynb`).
- Earlier timestamped files in `results/` from the same day (before 08:49 UTC) are debugging iterations (prompt-format fix, then repetition-penalty fix) — kept per the never-overwrite rule, superseded by `results_20261022T084947Z.csv`.
- Images used (GalaxyIDs 100008/100023/100053/100078/100090) are just the first 5 by ID — a pipeline smoke test, not the actual odd/ordinary research subset.
