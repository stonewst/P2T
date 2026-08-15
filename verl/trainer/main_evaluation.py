"""
Evaluate generated responses from main_generation.py using math-verify.

For each trajectory (response), compute correctness via math-verify, then:
  - Attach per-trajectory scores back to the parquet file.
  - Compute per-problem mean_acc.
  - Save a summary eval_results.json.

Usage:
    python main_evaluation.py \
        --input_path  /path/to/output.parquet \
        --output_dir  /path/to/eval_output \
        [--n_workers 32]

    # Or evaluate every *.parquet under a directory:
    python main_evaluation.py \
        --input_path  /path/to/dir/ \
        --output_dir  /path/to/eval_output \
        [--n_workers 32]
"""

import argparse
import ast
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_ground_truth(raw: Any) -> str:
    """Return a single ground-truth string from the stored reward_model value.

    main_generation.py stores ground_truth as either:
      - a plain string  -> use as-is
      - a str repr of a list, e.g. "['16', '16', ...]" -> take the first element
    """
    if isinstance(raw, list):
        return str(raw[0]) if raw else ""
    s = str(raw).strip()
    if s.startswith("["):
        try:
            lst = ast.literal_eval(s)
            return str(lst[0]) if lst else s
        except Exception:
            return s
    return s


def _score_one(response: str, ground_truth: str) -> float:
    """Call math-verify for a single (response, ground_truth) pair."""
    from verl.utils.reward_score.math_verify import compute_score  # local import for subprocess workers
    try:
        result = compute_score(response, ground_truth)
        return float(result["score"])
    except Exception:
        return 0.0


def _score_row_args(args):
    """Wrapper so ProcessPoolExecutor can pickle the call."""
    responses, ground_truth = args
    return [_score_one(r, ground_truth) for r in responses]


# ---------------------------------------------------------------------------
# core evaluation
# ---------------------------------------------------------------------------

def evaluate_parquet(input_path: str, output_dir: str, n_workers: int,
                     flush_every: int = 5) -> dict:
    """Evaluate one parquet file and return summary statistics.

    The JSON result file is written (overwritten) every ``flush_every`` problems
    as they finish scoring, so progress is not lost on an unexpected crash.
    """
    print(f"\n[evaluate] reading {input_path}")
    df = pd.read_parquet(input_path)

    assert "responses" in df.columns, "Column 'responses' not found – was this file produced by main_generation.py?"
    assert "reward_model" in df.columns, "Column 'reward_model' not found."

    n_problems = len(df)
    if n_problems == 0:
        print("[evaluate] no problems found, skipping.")
        return {"file": input_path, "n_problems": 0, "overall_mean_acc": 0.0}

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_json = os.path.join(output_dir, f"{base_name}__eval_results.json")

    # -----------------------------------------------------------------------
    # pre-build per-row metadata and scoring args in one pass
    # -----------------------------------------------------------------------
    row_meta = []
    task_args = []
    for _, row in df.iterrows():
        rm_raw = row["reward_model"]
        rm = rm_raw if isinstance(rm_raw, dict) else {}
        ei_raw = row.get("extra_info", None)
        ei = ei_raw if isinstance(ei_raw, dict) else {}
        responses = list(row["responses"])
        gt = _parse_ground_truth(rm.get("ground_truth", ei.get("gt_answer", "")))

        raw_prompt = row.get("prompt", [])
        if isinstance(raw_prompt, np.ndarray):
            prompt = raw_prompt.tolist()
        elif hasattr(raw_prompt, "tolist"):
            prompt = raw_prompt.tolist()
        else:
            prompt = list(raw_prompt) if raw_prompt is not None else []

        row_meta.append({
            "id_raw":        ei.get("id_raw", ""),
            "question":      ei.get("question", ""),
            "gt_answer":     gt,
            "prompt":        prompt,
            "n_samples":     len(responses),
            "response_list": responses,
        })
        task_args.append((responses, gt))

    # -----------------------------------------------------------------------
    # parallel scoring – build records on-the-fly, flush JSON every flush_every
    # -----------------------------------------------------------------------
    scores_per_problem: list[list[float]] = [None] * n_problems  # type: ignore[assignment]
    records: list[dict] = [None] * n_problems                    # type: ignore[assignment]
    completed_count = 0

    print(f"[evaluate] scoring {n_problems} problems × {len(task_args[0][0])} samples "
          f"with {n_workers} workers … (flush every {flush_every})")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {executor.submit(_score_row_args, args): i for i, args in enumerate(task_args)}
        for future in tqdm(as_completed(future_to_idx), total=n_problems, desc="scoring"):
            idx = future_to_idx[future]
            scores = future.result()
            scores_per_problem[idx] = scores
            records[idx] = {
                **row_meta[idx],
                "acc_list": scores,
                "mean_acc": float(np.mean(scores)),
            }
            completed_count += 1

            if completed_count % flush_every == 0:
                partial = [r for r in records if r is not None]
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(partial, f, ensure_ascii=False, indent=2)

    # final flush (covers tail when total % flush_every != 0)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([r for r in records if r is not None], f, ensure_ascii=False, indent=2)
    print(f"[evaluate] eval results saved to {out_json}")

    # -----------------------------------------------------------------------
    # attach results to the dataframe and save augmented parquet
    # -----------------------------------------------------------------------
    df["scores"] = scores_per_problem
    df["mean_acc"] = [float(np.mean(s)) for s in scores_per_problem]

    overall_mean_acc = float(df["mean_acc"].mean())
    print(f"[evaluate] overall mean_acc = {overall_mean_acc:.4f}")

    out_parquet = os.path.join(output_dir, f"{base_name}__eval.parquet")
    df.to_parquet(out_parquet)
    print(f"[evaluate] augmented parquet saved to {out_parquet}")

    return {"file": input_path, "n_problems": n_problems, "overall_mean_acc": overall_mean_acc}


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
# Output file structure:
# - train_split_1__eval.parquet: original data + new columns scores (list[float]) and mean_acc
# - train_split_1__eval_results.json: list, each element is a problem with fields:
#     id_raw / question / gt_answer / prompt / n_samples / response_list / acc_list / mean_acc
# - if evaluating multiple files, eval_results_all.json is also generated as a summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate generation outputs with math-verify.")
    parser.add_argument("--input_path", required=True,
                        help="Path to a .parquet file or a directory containing *.parquet files.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory where eval results will be saved.")
    parser.add_argument("--n_workers", type=int, default=32,
                        help="Number of parallel worker processes for scoring (default: 32).")
    args = parser.parse_args()

    # collect parquet files
    if os.path.isdir(args.input_path):
        parquet_files = sorted([
            os.path.join(args.input_path, f)
            for f in os.listdir(args.input_path)
            if f.endswith(".parquet")
        ])
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found under {args.input_path}")
    else:
        parquet_files = [args.input_path]

    all_summaries = []
    for pf in parquet_files:
        summary = evaluate_parquet(pf, args.output_dir, args.n_workers)
        all_summaries.append(summary)

    # if multiple files, print an aggregated overview
    if len(all_summaries) > 1:
        print("\n===== Aggregated Results =====")
        total_problems = sum(s["n_problems"] for s in all_summaries)
        weighted_acc = sum(s["overall_mean_acc"] * s["n_problems"] for s in all_summaries) / total_problems
        for s in all_summaries:
            print(f"  {os.path.basename(s['file']):<50s}  mean_acc={s['overall_mean_acc']:.4f}")
        print(f"  {'[overall weighted]':<50s}  mean_acc={weighted_acc:.4f}")

        merged_summary_path = os.path.join(args.output_dir, "eval_results_all.json")
        with open(merged_summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "files": all_summaries,
                "total_problems": total_problems,
                "overall_weighted_mean_acc": weighted_acc,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n[evaluate] merged summary saved to {merged_summary_path}")


if __name__ == "__main__":
    main()
