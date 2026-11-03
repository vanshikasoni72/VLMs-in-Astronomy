# Problem Statement & Research Plan — Diagnosing Why Vision Models Fail at Galaxy Anomaly Detection

**Date:** 2026-11-03
**Status:** Draft plan, built directly on the pilot results from Oct 22-23, 2026. Not yet executed at the scale described here.

---

## 1. Background

The pilot work (6 models, small samples, run on a local laptop) tested whether AI vision models can tell whether a galaxy looks ordinary or anomalous, checked against real Galaxy Zoo 2 (GZ2) volunteer votes. Two things came out of that pilot that are more interesting than a simple accuracy number:

- A model trained specifically on galaxy images (Zoobot) tracked human "odd" judgments noticeably better than two general-purpose models (CLIP, DINOv2) that have never seen astronomy data.
- Chat-style models that *describe* an image in words showed clear, distinct failure patterns — one invented specific wrong object names most of the time; another avoided the question almost entirely while still sounding confident.

These are **failure-mode findings**, not just pass/fail numbers. That's the basis for the revised direction below.

## 2. Problem statement

> **Why do general-purpose vision and vision-language models underperform at detecting anomalous galaxies, compared to a domain-specific model — and does that failure depend on the *type* of anomaly involved?**

This reframes the thesis from "can models detect anomalies" (a yes/no benchmarking question) to "where, specifically, do they fail, and why" (a diagnostic question). It is more tractable in the time available and builds directly on what the pilot already found.

## 3. Research questions

1. **RQ1 — Does the specialist-vs-general gap hold at scale, and is it uniform?** Does a domain-specific model's advantage over general-purpose models hold when tested on thousands of images rather than hundreds, and is the gap the same size for every kind of anomaly, or bigger for some types than others?
2. **RQ2 — Is the gap explained by *where* each model looks?** When a model's anomaly score is high, is it responding to the actual unusual feature in the image (a tidal tail, a ring, a dust lane), or to noise, artifacts, or irrelevant bright objects in the frame?
3. **RQ3 — Why do chat-style models hallucinate specific false identities?** Is this driven by the model defaulting to familiar, frequently-captioned objects (e.g. famous named galaxies) when uncertain, rather than genuinely reasoning about the image in front of it?
4. **RQ4 — How much of the "failure" is actually label noise?** Galaxy Zoo's vote fractions come from varying numbers of volunteers per galaxy; how much of the apparent model/human disagreement is actually unreliable ground truth rather than a real model error?

## 4. Models (4 chosen)

| # | Model | Role | Type | Why |
|---|---|---|---|---|
| 1 | **Zoobot** | Domain specialist | Embedding (CNN, ConvNeXt) | Trained specifically on Galaxy Zoo data; the pilot's best performer and the key comparison point for RQ1/RQ2. |
| 2 | **CLIP** | General baseline A | Embedding (ViT, text-supervised) | General-purpose, but trained with language supervision — tests whether text-grounding helps or hurts on astronomy images. |
| 3 | **DINOv2** | General baseline B | Embedding (ViT, self-supervised) | General-purpose with *no* language supervision at all — an independent check that the specialist-vs-general gap isn't just a CLIP quirk. |
| 4 | **Qwen2-VL (7B)** | Generative / language | Vision-language (chat) | The pilot's clearest hallucination case, now run at full scale (7B, not the 2B pilot version) on a real GPU — needed to properly test RQ3. |

**Why these four and not more:** Models 1-3 give a clean embedding comparison that supports both the taxonomy analysis (RQ1) and the attention/saliency analysis (RQ2) with standard, well-understood techniques. Model 4 is the one generative model needed to dig into the hallucination finding (RQ3) properly, rather than spreading effort thin across several chat models that would mostly just re-confirm "yes, they hallucinate" without explaining why. LLaVA-1.5-7B and a second BLIP-2 variant were tested in the pilot and can be added later as a replication check if time allows, but aren't required for the core research questions.

## 5. Data

### 5.1 What's already available

- Galaxy Zoo 2 / Kaggle "Galaxy Zoo - The Galaxy Challenge" dataset: **61,578 labeled galaxies**, already downloaded (training_solutions_rev1.csv + images_training_rev1.zip, ~830MB). No new download needed for the plan below.
- Real human labels already present in the data, not invented: `Class6.1` (any odd feature, yes-vote fraction) and `Class8.1`-`Class8.7` (which *kind* of odd feature, if any).

### 5.2 The anomaly-type breakdown (real counts, checked against the actual file)

| Odd-feature type | Column | Galaxies with vote fraction > 0.3 | Usable at scale? |
|---|---|---|---|
| Ring | Class8.1 | 1,869 | Yes |
| Merger | Class8.6 | 2,648 | Yes |
| Other (unclassified odd) | Class8.5 | 2,067 | Yes |
| Irregular | Class8.4 | 1,450 | Yes |
| Disturbed | Class8.3 | 307 | Thin but usable |
| Dust Lane | Class8.7 | 112 | Thin — treat results as exploratory |
| Lens or Arc | Class8.2 | 34 | **Too rare in GZ2 alone** — exclude from primary analysis or flag any finding here as unreliable |

Any galaxy with `Class6.1` > 0.5 (any kind of odd feature, strong vote): **8,463 galaxies** — this is the overall positive pool.

### 5.3 Proposed sample for this study

- **Category-positive sets:** all galaxies above the 0.3 threshold for the six usable categories above (~8,000 images before removing overlap — a galaxy can score high on more than one category).
- **Background/"ordinary" set:** 10,000 randomly drawn galaxies (not pre-filtered), for the embedding/outlier-detection method to have a realistic sense of what "typical" looks like, and as the main pool for RQ4's label-noise check.
- **Estimated total distinct images needed: ~15,000-18,000** out of the 61,578 available — well within the dataset already downloaded, no new data collection required.
- This is a large step up from the pilot's 2,000-image sample, and is the "decent scale" needed to say something statistically meaningful per category, especially for the thinner categories (Disturbed, Dust Lane).

### 5.4 What's *not* included (for now)

- The unlabeled Kaggle test set (`images_test_rev1.zip`) — no ground-truth votes, not usable for this study.
- Other surveys (DESI Legacy Imaging, Galaxy Zoo: DECaLS, etc.) — a natural follow-up for testing whether findings generalize beyond GZ2's specific telescope/era, but out of scope for the core plan. Worth flagging to your guide as a possible extension, not a requirement.

## 6. Method

### Part A — Taxonomy: does the specialist-vs-general gap depend on anomaly type? (RQ1)

1. For each of the 6 usable categories, extract embeddings (Zoobot, CLIP, DINOv2) and score anomaly-ness with Local Outlier Factor, same method as the pilot but now per category instead of pooled.
2. Compare each model's correlation with the real human vote, *per category* — the hypothesis is that the specialist model's advantage is bigger for subtle categories (Dust Lane, Ring) and smaller for visually dramatic ones (Merger), but this needs to be checked, not assumed.
3. For Qwen2-VL, check whether its description text actually mentions the relevant feature per category (e.g., does it mention "companion" or "interaction" for Merger-labeled images?).

### Part B — Mechanistic: is the model looking at the right place? (RQ2, RQ3)

1. For a handful of clear examples per category, generate a saliency map for each embedding model: attention-rollout for CLIP/DINOv2 (both transformer-based), Grad-CAM for Zoobot (CNN-based). Since these are embedding models with no classification head, the embedding vector's norm is used as the target signal.
2. For Qwen2-VL, extract cross-attention weights between the generated answer tokens and the image patches, to see what the model was "looking at" while writing its answer — particularly for the hallucination cases (does it attend to a real feature, or to noise, when it invents a specific wrong name?).
3. Overlay each saliency map on the actual image and assess by eye whether the highlighted region matches the real odd feature.

### Part B (continued) — Label-noise check (RQ4)

- Where Galaxy Zoo provides vote *counts* (not just fractions) for a subset of galaxies, restrict the "disagreement" analysis to high-confidence votes (many voters) and check whether model/human disagreement shrinks — isolating genuine model error from noisy ground truth.

## 7. Platforms

- **Kaggle** and **Google Colab** (free GPU tiers) for the actual model runs — both can mount the Kaggle dataset directly, no manual extraction needed like on the local laptop.
- Still searching for additional free compute if Part B's attention-extraction work (heavier, needs model internals) needs more GPU time than Kaggle/Colab's free quotas allow.
- Analysis, plotting, and the final write-up still happen locally, per the existing workflow.

## 8. What's already built and reusable

- `src/data.py`, `src/select_subset.py` — data loading and manifest-building, extendable to the category-based sampling above.
- `src/models.py` — adapters for Zoobot, CLIP, DINOv2, Qwen2-VL (2B tested; 7B needs a GPU session), BLIP-2 variants.
- `src/anomaly_lof.py` — the LOF-based anomaly scoring method, extendable to run per-category instead of pooled.
- `src/hallucination_rate.py` — proxy checks for named-object hallucination and task-engagement.
- None of Part B's saliency/attention code exists yet — that's new work.

## 9. Caveats

- This plan has not been run yet — it is a scaled-up design based on pilot findings, not itself a result.
- `Lens or Arc` is excluded from the primary analysis due to too few examples in GZ2 (34 at a reasonable confidence threshold).
- Vote fractions without counts (used in the pilot) are a known weakness; Section 6's label-noise check exists specifically to address this before trusting any "model failed here" conclusion.
- Four models is a deliberate scope limit to keep Part B (the new, harder mechanistic work) feasible — breadth was traded for depth.
