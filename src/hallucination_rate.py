"""Two cheap proxy checks on chat-VLM answers to the odd/ordinary prompt:

1. named_object_rate: does the answer confidently name a specific real
   astronomical object (NGC/IC/Messier/Sgr A*/Arp) for what is actually an
   anonymous Galaxy Zoo thumbnail? On its face, invented.
2. on_task_rate: does the answer even engage with the actual question
   (mentions odd/ordinary/anomalous/unusual/typical at all)?

Neither alone is the full picture — a model can score 0% on (1) while still
being unreliable, if it just ignores the question and states other kinds of
invented claims instead (see blip2_flant5's results: 0% named objects, but
only 10% of answers even address odd/ordinary).

Both are regex heuristics, not a manually verified fact-check of every claim.
"""
import argparse
import re

import pandas as pd

NAMED_OBJECT_PATTERN = re.compile(
    r"\b(NGC\s*\d+|IC\s*\d+|M\s?\d{1,3}\b|Messier\s*\d+|Sgr\s*A\*?|Sagittarius\s*A\*?|Arp\s*\d+)", re.IGNORECASE
)
ON_TASK_PATTERN = re.compile(r"\b(odd|ordinary|anomal\w*|unusual|typical)\b", re.IGNORECASE)


def rate(results_csv, model_name, pattern):
    df = pd.read_csv(results_csv)
    df = df[df["model_name"] == model_name]
    hits = df["answer"].apply(lambda a: bool(pattern.search(str(a))))
    return hits.sum(), len(df), df.loc[hits, "answer"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to a results CSV from run_eval.py")
    parser.add_argument("--model", required=True, help="model_name to check (e.g. blip2, qwen2vl_2b)")
    args = parser.parse_args()

    n_hits, n_total, examples = rate(args.results, args.model, NAMED_OBJECT_PATTERN)
    print(f"{args.model}: {n_hits}/{n_total} answers ({100 * n_hits / n_total:.0f}%) name a specific object")
    for a in examples.head(3):
        print("  example:", repr(str(a)[:100]))

    n_on_task, n_total, _ = rate(args.results, args.model, ON_TASK_PATTERN)
    print(f"{args.model}: {n_on_task}/{n_total} answers ({100 * n_on_task / n_total:.0f}%) even engage with odd/ordinary at all")


if __name__ == "__main__":
    main()
