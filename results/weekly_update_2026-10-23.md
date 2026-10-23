# Weekly Update — Oct 23, 2026

**Summary:** A small pilot test was carried out this week to confirm that AI models can be applied to real galaxy images and benchmarked against real human answers. Results below are preliminary, not final findings.

## What was done

- 6 AI models tested, using 2 different methods
- Both methods checked against real votes from Galaxy Zoo citizen-science volunteers (not invented answers)
- All testing run locally on a standard laptop (no GPU) — the largest models require more compute, see "Next steps"

## Method 1: ask the model directly, in words

"Does this galaxy look normal or odd?"

| Model | Makes up a wrong specific name | Actually answers the question |
|---|---|---|
| BLIP-2 (version A) | 10% | 76% |
| Qwen2-VL | 88% | 76% |
| BLIP-2 (version B) | 0% | 10% |

**Takeaway:** the model that "looks safest" (0% wrong names) is actually the *least* useful — it just avoids answering instead of being accurate. Sounding fluent does not mean being trustworthy.

## Method 2: score "unusualness" without asking in words

Each image is turned into a numeric fingerprint; distance from other images is measured and compared to real human votes.

| Model | Trained on | Matches human votes |
|---|---|---|
| Zoobot | galaxies specifically | **Best** |
| CLIP | general images + text | Middle |
| DINOv2 | general images | Lowest |

**Takeaway:** the model trained specifically on galaxies agrees with humans more than two different general-purpose models. A few examples were spot-checked by eye and the pattern holds up.

## Caveats

- 2 other astronomy-specific models skipped (needed GPU-only install, or unverified code)

## Next steps

- Run the 2 biggest models on Kaggle's free GPUs (too large for this laptop)
- Scale up to more images once direction is confirmed
