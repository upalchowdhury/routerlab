"""Print a Markdown table from a real benchmark JSON; never invent results."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    result = json.loads(args.input.read_text())
    print("GPU:", result["environment"]["gpu"])
    print("Mode:", result["measurement"]["mode"])
    print("Cache:", result["measurement"]["cache_regime"])
    print()
    print("| M | H | Experts | Backend | Median us | IQR us | Top-k set agreement |")
    print("|---:|---:|---:|---|---:|---:|---:|")
    for row in result["rows"]:
        if row["status"] != "passed":
            continue
        t = row["timing"]
        print(f"| {row['m']} | {row['h']} | {row['e']} | {row['backend']} | "
              f"{t['median_us']:.3f} | {t['p75_us'] - t['p25_us']:.3f} | "
              f"{row['numerics']['topk_set_match_fraction']:.4f} |")
    failures = [r for r in result["rows"] if r["status"] != "passed"]
    if failures:
        print("\nUnavailable or failed cases:")
        for row in failures:
            print(f"- {row['backend']} M={row['m']} H={row['h']} E={row['e']}: {row['error']}")
    print("\nProjection microbenchmark only. End-to-end inference improvement is unmeasured.")


if __name__ == "__main__":
    main()
