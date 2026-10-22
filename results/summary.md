# Results Summary

| date | model | prompt_id | n_images | results_file | headline | note |
|---|---|---|---|---|---|---|
| 2026-10-22 | zoobot | embedding_v1 | 5 | results_20261022T084947Z.csv | placeholder (embedding norm only) | Encoder-only (timm hub checkpoint); no GZ2 classification head wired up, so this is not a real morphology prediction yet. |
| 2026-10-22 | blip2 | p1 | 5 | results_20261022T084947Z.csv | real text answers | Needed `Question: ... Answer:` framing (it's a continuation model, not chat) plus a repetition penalty to stop it looping/echoing the prompt. |
| 2026-10-22 | qwen2vl_2b | p1 | 5 | results_20261022T084947Z.csv | real text answers | Confidently hallucinates specific object names (e.g. "NGC 1068", "Sgr A*") for anonymous Galaxy Zoo thumbnails — a real reliability finding, not a pipeline bug. |

**Notes (2026-10-22 run)**

- First true end-to-end run, on local CPU (no GPU available on this laptop). LLaVA-1.5-7B and Qwen2-VL-7B are still pending — too large to run here, queued for Kaggle (`notebooks/model_smoke_test.ipynb`).
- Earlier timestamped files in `results/` from the same day (before 08:49 UTC) are debugging iterations (prompt-format fix, then repetition-penalty fix) — kept per the never-overwrite rule, superseded by `results_20261022T084947Z.csv`.
- Images used (GalaxyIDs 100008/100023/100053/100078/100090) are just the first 5 by ID — a pipeline smoke test, not the actual odd/ordinary research subset.

## 2026-10-23: real odd/ordinary pilot + embedding anomaly detection

| date | model | prompt_id | n_images | results_file | headline | note |
|---|---|---|---|---|---|---|
| 2026-10-23 | blip2 | p1 | 50 | results_20261022T203649Z.csv | real text answers | 25 galaxies unanimously flagged "odd" (Class6.1=1.0) + 25 unanimously "ordinary" (Class6.2=1.0) — real GZ2 volunteer labels, not arbitrary images. Only **10% (5/50)** of answers name a specific object (NGC/Messier/etc.) for these anonymous thumbnails. |
| 2026-10-23 | qwen2vl_2b | p1 | 50 | results_20261022T214407Z.csv | real text answers | Same 50-image pilot. **88% (44/50)** of answers confidently name a specific object (e.g. "NGC 4569", "Andromeda Galaxy (M31)") that the image can't actually be — a much higher hallucination rate than BLIP-2 on identical inputs. |
| 2026-10-23 | zoobot | lof_v1 | 2000 | results_lof_20261022T224637Z.csv | Spearman r=0.300, p<0.0001 | Local Outlier Factor on Zoobot's (astronomy-specific) embeddings, vs. real human "odd" vote fraction (Class6.1), on a random (unfiltered) 2000-image sample. Stable 0.25-0.33 across n_neighbors=10/20/30. |
| 2026-10-23 | clip | lof_v1 | 2000 | results_lof_20261022T224637Z.csv | Spearman r=0.199, p<0.0001 | Same method, general-purpose CLIP (text-supervised) embeddings. |
| 2026-10-23 | dinov2 | lof_v1 | 2000 | results_lof_20261022T224637Z.csv | Spearman r=0.145, p<0.0001 | Same method, general-purpose DINOv2 (self-supervised) embeddings — a second, independent general baseline. |

**Headline findings for guide meeting**

1. **Domain-specific embeddings track human anomaly judgments better than general-purpose ones, and it's not a fluke of which general model you pick.** Zoobot (astronomy-specific) beats *both* CLIP and DINOv2 — two general-purpose models trained completely differently (text-supervised vs. self-supervised) — on the same 2000-image sample: r=0.300 vs. r=0.199 vs. r=0.145, all highly significant (p<0.0001), and the ranking is stable across three LOF neighborhood sizes (10/20/30). An earlier, noisier 300-image pilot found the same ordering with a wider gap (r=0.403 vs. r=0.126). Plot: `results/lof_vs_human_odd_votes.png` (all 3 models, full 2000-image sample).
2. **Bigger/more fluent ≠ more reliable.** Qwen2-VL-2B writes more detailed, confident answers than BLIP-2, but hallucinates a specific (wrong) object identity 88% of the time vs. BLIP-2's 10%, on identical anonymous thumbnails. Directly relevant to "can VLMs be *trusted* for anomaly detection", not just "can they describe images."

**Methodology notes / caveats**

- Odd/ordinary labels are the actual GZ2 volunteer vote fractions (`Class6.1`="is there anything odd? yes", `Class6.2`="no"), from `training_solutions_rev1.csv` — not an invented heuristic. Vote *counts* aren't in this file (only fractions), so a 1.0 fraction from few voters is treated the same as one from many; worth revisiting for the real research subset.
- The LOF sample is a plain random draw (unfiltered), so the correlation isn't biased by cherry-picking extremes — the 50-image VLM pilot *is* the extremes (most-unanimous odd/ordinary), chosen for interpretable qualitative examples, not statistical rigor.
- Hallucination-rate check is a simple regex for named-object patterns (NGC/IC/Messier/Sgr A*/Arp) in the answer text — a proxy, not a manually verified fact-check of each claim.
- Still pending: LLaVA-1.5-7B, Qwen2-VL-7B (Kaggle GPU only — too large for this laptop's 15GB RAM). AstroCLIP/AstroDINO were investigated but need either `trust_remote_code` on an unverified community mirror or a CUDA-only install — skipped for now rather than taking either risk silently.
