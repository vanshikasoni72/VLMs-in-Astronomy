# Guide update — 2026-10-23: first real engagement with models

## What this covers

Six models, two complementary approaches to "can a model spot an anomalous galaxy", both checked against **real Galaxy Zoo 2 volunteer labels** (not invented ground truth):

- **Chat-style VLMs** (BLIP-2 with an OPT decoder, BLIP-2 with a Flan-T5-XL decoder, Qwen2-VL-2B): asked directly, in text, whether a galaxy looks ordinary or odd.
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

A couple of concrete examples, checked against the actual images (not just the numbers):
- Galaxy `588555`: Zoobot's #1 most-anomalous-by-embedding galaxy in the sample, and volunteers rated it 81% odd. Looking at the image: a chaotic, clumpy structure with no clear symmetric shape — it genuinely does look odd. Model and humans agree, and agree correctly.
- Galaxy `202237`: volunteers rated it 88% odd, but Zoobot's embedding ranks it in the bottom ~18% for anomaly score. Looking at the image: it's actually a fairly smooth, ordinary-looking round galaxy — nothing visually jumps out. This looks less like a model failure and more like a noisy human label (see the vote-count caveat below) — worth keeping in mind before assuming every "disagreement" means the model is wrong.

## Finding 2: more fluent isn't more trustworthy — and one metric alone can mislead

Same 50-image pilot (25 galaxies volunteers unanimously flagged odd, 25 unanimously ordinary) asked to three text models — including two BLIP-2 variants that share the same vision encoder but differ only in decoder (plain OPT vs. instruction-tuned Flan-T5-XL):

| Model | Names a specific (wrong) object for an anonymous thumbnail | Actually engages with "odd vs. ordinary" at all |
|---|---|---|
| BLIP-2 (OPT decoder) | 10% (5/50) | 76% (38/50) |
| Qwen2-VL-2B | **88% (44/50)** | 76% (38/50) |
| BLIP-2 (Flan-T5-XL decoder) | **0% (0/50)** | **10% (5/50)** |

Two things worth saying out loud:

1. Qwen2-VL-2B's answers read as more confident and detailed, but it's dramatically more likely to confidently assert a specific (impossible) object identity — e.g. calling an anonymous Galaxy Zoo thumbnail "the Andromeda Galaxy (M31)". Checking one case by eye (galaxy `516751`): it called it "the galaxy NGC 4569 (also known as Arp 271)". The actual image genuinely is an interacting galaxy pair with a visible tidal tail — so the model's *structural read* was roughly right, but the *specific invented identity* was not. That's a more precise, more concerning failure mode than "hallucinates randomly": it senses something real, then confidently fabricates false specifics on top of it.
2. The more instruction-tuned Flan-T5-XL decoder has the *best* score on the named-object metric (0%) — but it's actually the **least reliable of the three**: it barely ever engages with the actual question, mostly dodging into confidently-stated invented pseudo-facts instead (e.g. "called a flare galaxy", shaped "like a wormhole" or "a tetrahedron"). A single hallucination metric, taken alone, would have ranked this model best. It isn't. Worth flagging as a methodology lesson as much as a model-comparison one.

## Caveats (worth saying out loud to your guide)

- This is a pipeline-and-signal check, not the actual research subset or final method — image counts, LOF hyperparameters, and the prompt wording are all provisional.
- `Class6.1`/`Class6.2` are vote *fractions* only (no vote counts available in this file), so a unanimous 1.0 from 2 voters counts the same as one from 50.
- The hallucination checks are regex proxies (names a specific object / mentions odd-ordinary-anomalous-unusual-typical), not a manual fact-check of every claim.
- Zoobot/CLIP/DINOv2 here are used as **embeddings only** — no classification head — so "LOF anomaly score" is a general outlier-detection signal, not a GZ2-style morphology prediction.
- AstroCLIP and AstroDINO (official astronomy-specific alternatives) were investigated but skipped: AstroCLIP's real install is CUDA-only, and the only other public weights need running unverified remote code — not worth the risk for a pilot.

## Next steps

- Run LLaVA-1.5-7B and Qwen2-VL-7B on Kaggle (`notebooks/model_smoke_test.ipynb`) to complete the planned 5-ish model lineup with the larger chat models.
- Decide the actual research angle: pure LOF/embedding anomaly detection, VLM-as-judge, or a hybrid — this pilot suggests embeddings + a specialist model is the more defensible starting point.
- If continuing the embedding route, look at *why* Zoobot's embedding space tracks human judgment better — which features/examples drive it, and where it fails (like galaxy `202237` above).
