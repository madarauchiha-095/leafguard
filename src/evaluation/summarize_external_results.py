"""Write compact CSV/Markdown summaries and conditional intervals for an external run."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.inference.common import CLASS_NAMES, MODEL_SPECS, ROOT


def stratified_intervals(matrix, mapped_names, rng, replicates=10_000):
    matrix = np.asarray(matrix, dtype=np.int64)
    class_indices = [CLASS_NAMES.index(name) for name in mapped_names]
    rows = matrix[class_indices, :]
    supports = rows.sum(axis=1)
    if np.any(supports <= 0):
        raise ValueError("Every mapped true class must have positive support")
    draws = np.stack([
        rng.multinomial(int(count), row / count, size=replicates)
        for row, count in zip(rows, supports)
    ], axis=1)
    prediction_counts = draws.sum(axis=1)
    true_positives = np.stack([
        draws[:, i, class_index]
        for i, class_index in enumerate(class_indices)
    ], axis=1)
    mapped_prediction_counts = prediction_counts[:, class_indices]
    f1 = 2 * true_positives / np.maximum(
        supports[None, :] + mapped_prediction_counts, 1
    )
    accuracy = true_positives.sum(axis=1) / supports.sum()
    macro_f1 = f1.mean(axis=1)
    return {
        "accuracy_95_percentile_ci": np.percentile(accuracy, [2.5, 97.5]).tolist(),
        "macro_f1_95_percentile_ci": np.percentile(macro_f1, [2.5, 97.5]).tolist(),
        "replicates": replicates,
        "method": "Stratified image bootstrap from confusion-matrix rows; resample outcomes within each observed true class while preserving class supports.",
        "interpretation_limit": "Conditional on the selected source split, mapped samples, and observed supports; does not capture dataset-selection or mapping uncertainty.",
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--dataset-slug", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")

    summary_rows, class_rows, intervals = [], [], {
        "dataset": args.dataset_name,
        "seed": args.seed,
        "models": {},
    }
    rng = np.random.default_rng(args.seed)
    for model_key, spec in MODEL_SPECS.items():
        path = args.results_dir / f"{model_key}_{args.dataset_slug}_test_results.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result["metrics"]
        confidence_records = json.loads(
            (args.results_dir / f"{args.dataset_slug}_predictions.json").read_text(encoding="utf-8")
        )
        model_confidence = [row["confidence"] for row in confidence_records if row["model"] == model_key]
        summary_rows.append({
            "model": spec["name"],
            "model_key": model_key,
            "samples": metrics["total"],
            "mapped_classes": len(result["mapped_classes"]),
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "mean_max_softmax_confidence": float(np.mean(model_confidence)),
            "correct": metrics["correct"],
            "total_inference_seconds": metrics["total_inference_seconds"],
        })
        for label, values in result["per_class"].items():
            class_rows.append({
                "model": spec["name"],
                "model_key": model_key,
                "class": label,
                "support": values["support"],
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1-score"],
            })
        intervals["models"][model_key] = {
            "mapped_classes": result["mapped_classes"],
            "sample_count": metrics["total"],
            **stratified_intervals(
                result["confusion_matrix"], result["mapped_classes"], rng,
                args.bootstrap_replicates,
            ),
        }

    write_csv(args.results_dir / f"{args.dataset_slug}_metrics.csv", summary_rows, list(summary_rows[0]))
    write_csv(args.results_dir / f"{args.dataset_slug}_per_class_metrics.csv", class_rows, list(class_rows[0]))
    (args.results_dir / f"{args.dataset_slug}_stratified_intervals.json").write_text(
        json.dumps(intervals, indent=2), encoding="utf-8"
    )

    percentage_columns = ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")
    lines = [
        f"# {args.dataset_name}: frozen-checkpoint evaluation",
        "",
        f"All four existing checkpoints were evaluated on the same {summary_rows[0]['samples']} mapped test images. No training or fine-tuning was performed.",
        "",
        "| Model | n | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 | Mean max-softmax |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        values = [f"{row[name] * 100:.2f}%" for name in percentage_columns]
        lines.append(
            f"| {row['model']} | {row['samples']} | {values[0]} | {values[1]} | {values[2]} | {values[3]} | {values[4]} | {row['mean_max_softmax_confidence']:.3f} |"
        )
    lines.extend([
        "",
        "Macro metrics average over mapped ground-truth classes. Predictions into any of the ten model output classes count as incorrect when they do not match the true class. Maximum softmax is a score, not calibrated uncertainty.",
        "",
        "Per-class results and sample supports are in the companion CSV. Percentile intervals are conditional stratified image-bootstrap intervals; they do not account for source selection or mapping uncertainty.",
        "",
        "| Model | Accuracy 95% CI | Macro F1 95% CI |",
        "|---|---:|---:|",
    ])
    for row in summary_rows:
        interval = intervals["models"][row["model_key"]]
        acc = interval["accuracy_95_percentile_ci"]
        f1 = interval["macro_f1_95_percentile_ci"]
        lines.append(
            f"| {row['model']} | [{acc[0]*100:.2f}%, {acc[1]*100:.2f}%] | [{f1[0]*100:.2f}%, {f1[1]*100:.2f}%] |"
        )
    lines.extend([
        "",
        "The sample is a mapped subset of the official test split. Unmapped labels, excluded licenses, and duplicate-screening decisions are detailed in `research/external_dataset_review.md` and `results/audit/plantseg_v7_archive_audit.json`.",
        "",
    ])
    (args.results_dir / f"{args.dataset_slug}_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote summary CSV, per-class CSV, Markdown, and bootstrap intervals under {args.results_dir}")


if __name__ == "__main__":
    main()
