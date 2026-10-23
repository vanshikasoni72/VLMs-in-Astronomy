# What we tried, and what we found (Oct 23, 2026)

## The idea

The big question for this thesis is whether AI models can look at a galaxy picture and tell if it looks normal or unusual. Before starting the real research, I ran a small trial to check the basic idea works and to see how different models behave.

I tested six AI models, two different ways, and checked both against real votes from Galaxy Zoo's citizen-science volunteers, who've already looked at these pictures and voted on whether each galaxy looks "odd." That gives real, trustworthy answers to check the AI against, instead of guessing.

## Way 1: asking the AI directly, in words

Three "chat" models were shown a galaxy picture and asked, in plain English, whether it looks normal or odd.

Fluent-sounding answers were not the same as trustworthy ones. One model invented a specific wrong fact in 88 of 100 answers — confidently naming the picture as a famous real galaxy (e.g. "the Andromeda Galaxy"), when it was actually just an anonymous, uncatalogued one. A simpler model did this far less (about 1 in 10).

A more "advanced" version of that simpler model stopped inventing wrong names almost entirely — but only because it barely answered the question at all. Instead it filled its answers with other made-up claims, like calling something "a flare galaxy" (not a real term) or describing a shape "like a wormhole." On paper it looked safest (zero wrong names), but it was actually the least useful of the three, since it wasn't really doing the task. Lesson: one simple check isn't enough — you have to check several things, and check them against the real images, not just trust a single number.

## Way 2: measuring "unusualness" without words

The second method skips asking a question entirely. Every AI model can turn an image into a kind of numerical fingerprint, where similar-looking images get similar fingerprints and unusual ones sit apart from the rest. A standard statistical method measures exactly how far apart each image sits from its neighbours — that's the "unusualness score" used here.

I tried this with three models on 2,000 galaxy pictures: one trained specifically on galaxy images, and two general-purpose models that have never seen astronomy data. Then I checked whether each model's unusualness score actually lined up with the real human votes.

The specialist model (trained on galaxies) agreed with the real human votes noticeably more than either general-purpose model, and this held up even after changing some technical settings — not a fluke. Checking a couple of examples by eye: the galaxy the model flagged as most unusual really does look chaotic and strange in the picture, matching the humans. One galaxy humans called "odd" actually looked fairly ordinary to me too — a useful reminder that human votes aren't perfect either, since a picture with few voters can look just as "certain" as one with many.

## Why this matters

This trial shows the full pipeline works: pulling in real galaxy pictures, pulling in real human judgements to check against, running several different AI models on the same data, and automatically comparing results to the truth. Every number above came from an actual run, and I spot-checked several results against the real images myself — nothing here is guessed.

Because it works at this small scale, the sensible next step is the same pipeline on a much bigger platform (free cloud GPUs) with more images, and with larger, more capable versions of these same models that are too big to run on a laptop.

## Worth saying plainly

- This is a trial to prove the idea works, not final results — exact numbers will change once run properly at scale.
- Human "odd" votes are percentages, not raw counts, so a galaxy with only a few voters can look just as certain as one with many. Worth fixing before the real run.
- Two other astronomy-specific models exist but weren't used: one needs specialised hardware we don't have, the other needs running code from an unverified source — not worth that risk for a quick trial.
