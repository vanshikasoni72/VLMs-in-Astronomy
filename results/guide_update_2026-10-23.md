# Guide update — 2026-10-23: first real engagement with models

## What this covers

Five models, two complementary approaches to "can a model spot an anomalous galaxy", both checked against **real Galaxy Zoo 2 volunteer labels** (not invented ground truth):

- **Chat-style VLMs** (BLIP-2, Qwen2-VL-2B): asked directly, in text, whether a galaxy looks ordinary or odd.
- **Embedding-based anomaly detection** (Zoobot, CLIP, DINOv2): extract each model's image embedding, score outlier-ness with **Local Outlier Factor** (a standard, established novelty-detection method), and check whether that score tracks what volunteers actually said.

Everything ran locally (no GPU on this laptop) — LLaVA-1.5-7B and Qwen2-VL-7B are still pending on Kaggle, since they're too large to run here.

## Finding 1: specialist embeddings beat general-purpose ones

Galaxy Zoo 2 volunteers vote on "is there anything odd about this galaxy?" (`Class6.1` = yes-fraction). On a random sample of 2000 galaxies (not cherry-picked), Local Outlier Factor scores from each model's embeddings correlate with that real human judgment as follows:

| Model | Type | Spearman r vs. human "odd" vote | p-value |
|---|---|---|---|
| **Zoobot** | astronomy-specific (trained on GZ data) | **0.300** | <0.0001 |
| CLIP | general, text-supervised | 0.199 | <0.0001 |
| DINOv2 | general, self-supervised | 0.145 | <0.0001 |

Zoobot wins against *two* differently-trained general-purpose baselines, and the ranking is stable whether the LOF neighborhood size is 10, 20, or 30 — so it's not a fluke of one hyperparameter choice. See the plot (`lof_vs_human_odd_votes.png`, sent separately).

A couple of concrete examples for discussion:
- Galaxy `588555`: Zoobot's #1 most-anomalous-by-embedding galaxy in the sample, and volunteers rated it 81% odd — a clean hit.
- Galaxy `202237`: volunteers rated it 88% odd, but Zoobot's embedding ranks it in the bottom ~18% for anomaly score — a clear miss, worth a look at the actual image to understand what the embedding is missing.

## Finding 2: more fluent isn't more trustworthy

Same 50-image pilot (25 galaxies volunteers unanimously flagged odd, 25 unanimously ordinary) asked to both BLIP-2 and Qwen2-VL-2B in text:

| Model | Hallucinates a specific object name (NGC/Messier/etc.) for an anonymous thumbnail |
|---|---|
| BLIP-2 | 10% (5/50) |
| Qwen2-VL-2B | **88% (44/50)** |

Qwen2-VL-2B's answers read as more confident and detailed, but it's dramatically more likely to confidently assert a specific (impossible) object identity — e.g. calling an anonymous Galaxy Zoo thumbnail "the Andromeda Galaxy (M31)". That's a direct hit on whether a model can be *trusted* for anomaly detection, not just whether it can describe an image.

## Caveats (worth saying out loud to your guide)

- This is a pipeline-and-signal check, not the actual research subset or final method — image counts, LOF hyperparameters, and the prompt wording are all provisional.
- `Class6.1`/`Class6.2` are vote *fractions* only (no vote counts available in this file), so a unanimous 1.0 from 2 voters counts the same as one from 50.
- The hallucination check is a regex proxy (does the answer name a specific object), not a manual fact-check of every claim.
- Zoobot/CLIP/DINOv2 here are used as **embeddings only** — no classification head — so "LOF anomaly score" is a general outlier-detection signal, not a GZ2-style morphology prediction.
- AstroCLIP and AstroDINO (official astronomy-specific alternatives) were investigated but skipped: AstroCLIP's real install is CUDA-only, and the only other public weights need running unverified remote code — not worth the risk for a pilot.

## Next steps

- Run LLaVA-1.5-7B and Qwen2-VL-7B on Kaggle (`notebooks/model_smoke_test.ipynb`) to complete the planned 5-ish model lineup with the larger chat models.
- Decide the actual research angle: pure LOF/embedding anomaly detection, VLM-as-judge, or a hybrid — this pilot suggests embeddings + a specialist model is the more defensible starting point.
- If continuing the embedding route, look at *why* Zoobot's embedding space tracks human judgment better — which features/examples drive it, and where it fails (like galaxy `202237` above).
