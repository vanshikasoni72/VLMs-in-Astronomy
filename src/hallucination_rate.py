"""Quick proxy metric: how often does a model's answer confidently name a
specific astronomical object (NGC/IC/Messier/Sgr A*/Arp) for what is
actually an anonymous Galaxy Zoo thumbnail?

This is a regex heuristic, not a manually verified fact-check of every
claim — but naming a specific object for an unlabeled SDSS cutout is, on
its face, invented, since nothing in the input identifies it as a known
object.
"""
import argparse
import re

import pandas as pd

PATTERN = re.compile(r"\b(NGC\s*\d+|IC\s*\d+|M\s?\d{1,3}\b|Messier\s*\d+|Sgr\s*A\*?|Sagittarius\s*A\*?|Arp\s*\d+)", re.IGNORECASE)


def rate(results_csv, model_name):
    df = pd.read_csv(results_csv)
    df = df[df["model_name"] == model_name]
    hits = df["answer"].apply(lambda a: bool(PATTERN.search(str(a))))
    return hits.sum(), len(df), df.loc[hits, "answer"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to a results CSV from run_eval.py")
    parser.add_argument("--model", required=True, help="model_name to check (e.g. blip2, qwen2vl_2b)")
    args = parser.parse_args()

    n_hits, n_total, examples = rate(args.results, args.model)
    print(f"{args.model}: {n_hits}/{n_total} answers ({100 * n_hits / n_total:.0f}%) name a specific object")
    for a in examples.head(3):
        print("  example:", repr(str(a)[:100]))


if __name__ == "__main__":
    main()
