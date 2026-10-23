# Weekly Update — Oct 23, 2026

**This week:** Ran a small test (not final results) to check that I can actually get AI models working on real galaxy pictures, and compare them against real human answers.

## What I did

- Tested **6 AI models**, 2 different ways
- Checked both ways against real votes from Galaxy Zoo citizen-science volunteers (not invented answers)
- Everything ran on my laptop (no GPU) — the biggest models need a bigger computer, see "Next steps"

## Method 1: ask the model directly, in words

"Does this galaxy look normal or odd?"

| Model | Makes up a wrong specific name | Actually answers the question |
|---|---|---|
| BLIP-2 (version A) | 10% | 76% |
| Qwen2-VL | 88% | 76% |
| BLIP-2 (version B) | 0% | 10% |

**Takeaway:** the model that "looks safest" (0% wrong names) is actually the *least* useful — it just avoids answering instead of being accurate. Sounding fluent does not mean being trustworthy.

## Method 2: score "unusualness" without asking in words

Turn each image into a numeric fingerprint, measure how far it sits from other images, compare that score to real human votes.

| Model | Trained on | Matches human votes |
|---|---|---|
| Zoobot | galaxies specifically | **Best** |
| CLIP | general images + text | Middle |
| DINOv2 | general images | Lowest |

**Takeaway:** the model trained specifically on galaxies agrees with humans more than two different general-purpose models. Spot-checked a few examples by eye — holds up.

## Caveats

- Small trial — numbers will change at real scale
- Human "odd" votes are percentages, not vote counts, so some may be noisy
- 2 other astronomy-specific models skipped (needed GPU-only install, or unverified code)

## Next steps

- Run the 2 biggest models on Kaggle's free GPUs (too large for this laptop)
- Decide final research direction based on which method looks more promising
- Scale up to more images once direction is confirmed
