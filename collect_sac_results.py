#!/usr/bin/env python3
"""Collect SAC training/evaluation data for 2026-02-11 and 2026-02-12 runs."""
import os
import csv
import glob
from pathlib import Path

BASE = Path("/workspaces/sinergym")
patterns = [
    "Eplus-SAC-training-nuestroMultizona_2026-02-11*",
    "Eplus-SAC-training-nuestroMultizona_2026-02-12*",
]

def get_final_training_metrics(progress_path):
    """Return last meaningful row of progress.csv (skip header and all-zero rows)."""
    if not progress_path.exists():
        return None
    rows = list(csv.reader(open(progress_path)))
    if len(rows) < 2:
        return None
    header = rows[0]
    # Find last row with non-zero mean_reward (column index 2)
    for i in range(len(rows) - 1, 0, -1):
        r = rows[i]
        if len(r) > 2:
            try:
                val = float(r[2])  # mean_reward
                if val != 0.0 or (len(r) > 4 and float(r[4]) != 0):  # or comfort term
                    return {"header": header, "row": r, "episode": r[0] if r else None}
            except (ValueError, IndexError):
                pass
    # Fallback: second-to-last row (last is often 0,0,0)
    if len(rows) >= 2:
        return {"header": header, "row": rows[-2], "episode": rows[-2][0]}
    return None

def read_evaluation_metrics(csv_path):
    """Return full contents of evaluation_metrics.csv as list of dicts."""
    if not csv_path.exists():
        return None
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return list(reader)

def main():
    all_dirs = []
    for p in patterns:
        all_dirs.extend(BASE.glob(p))
    all_dirs = sorted(set(d.name for d in all_dirs))

    training_dirs = [d for d in all_dirs if "_EVALUATION" not in d]
    evaluation_dirs = [d for d in all_dirs if "_EVALUATION" in d]

    results = []

    for dname in training_dirs:
        dpath = BASE / dname
        # Timestamp from dir name: e.g. 2026-02-11_22-29
        parts = dname.replace("Eplus-SAC-training-nuestroMultizona_", "").replace("-res1", "")
        timestamp = parts

        has_eval_csv = (dpath / "evaluation" / "evaluation_metrics.csv").exists()
        has_best_model = (dpath / "evaluation" / "best_model.zip").exists()

        progress = get_final_training_metrics(dpath / "progress.csv")
        final_reward = None
        if progress:
            row = progress["row"]
            if len(row) > 1:
                try:
                    final_reward = float(row[1])  # mean_reward (column index 1)
                except ValueError:
                    pass

        eval_metrics = None
        eval_path = dpath / "evaluation" / "evaluation_metrics.csv"
        if eval_path.exists():
            eval_metrics = read_evaluation_metrics(eval_path)

        results.append({
            "type": "TRAINING",
            "directory": dname,
            "timestamp": timestamp,
            "evaluation/evaluation_metrics.csv": has_eval_csv,
            "evaluation/best_model.zip": has_best_model,
            "final_training_mean_reward": final_reward,
            "progress_last_row": progress["row"] if progress else None,
            "evaluation_metrics": eval_metrics,
        })

    # EVALUATION directories (separate runs, may have their own progress or metrics)
    for dname in evaluation_dirs:
        dpath = BASE / dname
        parts = dname.replace("Eplus-SAC-training-nuestroMultizona_", "").replace("_EVALUATION-res1", "")
        timestamp = parts

        # Evaluation dirs might have evaluation_metrics in evaluation/ or root
        eval_path1 = dpath / "evaluation" / "evaluation_metrics.csv"
        eval_path2 = dpath / "evaluation_metrics.csv"
        has_eval_csv = eval_path1.exists() or eval_path2.exists()
        eval_path = eval_path1 if eval_path1.exists() else eval_path2
        has_best_model = (dpath / "evaluation" / "best_model.zip").exists() or (dpath / "best_model.zip").exists()

        progress = get_final_training_metrics(dpath / "progress.csv")
        final_reward = None
        if progress and len(progress["row"]) > 2:
            try:
                final_reward = float(progress["row"][2])
            except ValueError:
                pass

        eval_metrics = read_evaluation_metrics(eval_path) if eval_path.exists() else None

        results.append({
            "type": "EVALUATION",
            "directory": dname,
            "timestamp": timestamp,
            "evaluation/evaluation_metrics.csv": has_eval_csv,
            "evaluation/best_model.zip": has_best_model,
            "final_training_mean_reward": final_reward,
            "progress_last_row": progress["row"] if progress else None,
            "evaluation_metrics": eval_metrics,
        })

    # Print structured output
    for r in results:
        print("\n" + "=" * 80)
        print("DIRECTORY:", r["directory"])
        print("TYPE:", r["type"])
        print("TIMESTAMP:", r["timestamp"])
        print("evaluation/evaluation_metrics.csv:", r["evaluation/evaluation_metrics.csv"])
        print("evaluation/best_model.zip:", r["evaluation/best_model.zip"])
        print("FINAL TRAINING MEAN REWARD:", r["final_training_mean_reward"])
        if r["progress_last_row"]:
            print("PROGRESS LAST ROW (episode, mean_reward, ...):", r["progress_last_row"][:5], "..." if len(r["progress_last_row"]) > 5 else "")
        if r["evaluation_metrics"]:
            print("EVALUATION METRICS (full):")
            for i, row in enumerate(r["evaluation_metrics"]):
                print("  Episode", i, ":", dict(row))
        else:
            print("EVALUATION METRICS: (none)")
    print("\n" + "=" * 80)
    print("SUMMARY: %d TRAINING dirs, %d EVALUATION dirs" % (len(training_dirs), len(evaluation_dirs)))

if __name__ == "__main__":
    main()
