"""Summarize per-image softmax confidence and failures for an external evaluation."""
import argparse
import csv
import json
import itertools
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import binomtest

from src.inference.common import ROOT


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "quantiles": {}}
    qs = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "quantiles": {key: float(value) for key, value in zip(
            ("p10", "p25", "p50", "p75", "p90"), qs
        )},
    }


def load_predictions(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["confidence"] = float(row["confidence"])
        row["correct"] = row["correct"].strip().lower() == "true"
    return rows


def analyze_model(model_key, rows, external_result, pv_result, threshold, top_n):
    correct = [r["confidence"] for r in rows if r["correct"]]
    incorrect = [r["confidence"] for r in rows if not r["correct"]]
    confident = [r for r in rows if r["confidence"] >= threshold]
    confident_errors = [r for r in confident if not r["correct"]]
    confusion_counts = Counter(
        (r["true_label"], r["predicted_label"]) for r in rows if not r["correct"]
    )
    top_confusions = [
        {"true_label": pair[0], "predicted_label": pair[1], "count": count}
        for pair, count in confusion_counts.most_common(top_n)
    ]
    per_class = external_result["per_class"]
    pv_per_class = pv_result["per_class"]
    degradations = {}
    for label, ext in per_class.items():
        pv = pv_per_class.get(label)
        if pv is None:
            continue
        pv_f1, ext_f1 = float(pv["f1-score"]), float(ext["f1-score"])
        degradations[label] = {
            "plantvillage_f1": pv_f1,
            "external_f1": ext_f1,
            "f1_change_external_minus_plantvillage": ext_f1 - pv_f1,
            "f1_degradation_percentage_points": (pv_f1 - ext_f1) * 100,
            "plantvillage_support": int(pv["support"]),
            "external_support": int(ext["support"]),
        }
    worst_class = min(per_class, key=lambda label: per_class[label]["f1-score"])
    return {
        "model_name": external_result["model"],
        "sample_count": len(rows),
        "correct_count": len(correct),
        "incorrect_count": len(incorrect),
        "confidence_is_maximum_softmax_probability_not_calibrated_uncertainty": True,
        "confidence_by_outcome": {
            "correct": distribution(correct),
            "incorrect": distribution(incorrect),
        },
        "high_confidence_error_summary": {
            "threshold": threshold,
            "predictions_at_or_above_threshold": len(confident),
            "incorrect_at_or_above_threshold": len(confident_errors),
            "error_fraction_among_predictions_at_or_above_threshold": (
                len(confident_errors) / len(confident) if confident else None
            ),
            "incorrect_fraction_of_all_predictions": len(confident_errors) / len(rows) if rows else None,
        },
        "highest_confidence_incorrect": sorted(
            [r for r in rows if not r["correct"]], key=lambda r: r["confidence"], reverse=True
        )[:top_n],
        "lowest_confidence_correct": sorted(
            [r for r in rows if r["correct"]], key=lambda r: r["confidence"]
        )[:top_n],
        "most_common_confusion_pairs": top_confusions,
        "worst_external_class_by_f1": {
            "label": worst_class,
            "f1": float(per_class[worst_class]["f1-score"]),
            "support": int(per_class[worst_class]["support"]),
        },
        "per_class_plantvillage_to_external": degradations,
    }


def paired_comparisons(grouped, bootstrap_replicates, seed):
    model_keys = list(grouped)
    image_sets = [{row["image_id"] for row in grouped[key]} for key in model_keys]
    if any(ids != image_sets[0] for ids in image_sets[1:]):
        raise ValueError("Paired comparison requires the same image IDs for every model")
    by_id = {
        key: {row["image_id"]: row for row in grouped[key]}
        for key in model_keys
    }
    image_ids = sorted(image_sets[0])
    truth_by_model = [
        [by_id[key][image_id]["true_label"] for image_id in image_ids]
        for key in model_keys
    ]
    if any(truth != truth_by_model[0] for truth in truth_by_model[1:]):
        raise ValueError("True labels differ between model prediction records")
    labels = np.asarray(truth_by_model[0], dtype=object)
    classes = sorted(set(labels.tolist()))
    predictions = {
        key: np.asarray([by_id[key][image_id]["predicted_label"] for image_id in image_ids], dtype=object)
        for key in model_keys
    }

    # Resample images jointly across models, stratified by observed true class.
    rng = np.random.default_rng(seed)
    n = len(image_ids)
    weights = np.zeros((bootstrap_replicates, n), dtype=np.int16)
    for label in classes:
        indices = np.flatnonzero(labels == label)
        draws = rng.choice(indices, size=(bootstrap_replicates, len(indices)), replace=True)
        weights[:, indices] = (draws[:, :, None] == indices[None, None, :]).sum(axis=1)

    accuracy_replicates = {}
    macro_f1_replicates = {}
    observed_accuracy = {}
    observed_macro_f1 = {}
    class_indices = {label: np.flatnonzero(labels == label) for label in classes}
    for key in model_keys:
        pred = predictions[key]
        correct = (pred == labels).astype(np.float64)
        observed_accuracy[key] = float(correct.mean())
        accuracy_replicates[key] = (weights @ correct) / n
        class_f1 = []
        for label in classes:
            true_indices = class_indices[label]
            true_positive = weights[:, true_indices] @ (pred[true_indices] == label).astype(np.float64)
            predicted_count = weights @ (pred == label).astype(np.float64)
            denominator = len(true_indices) + predicted_count
            class_f1.append(np.divide(
                2 * true_positive, denominator,
                out=np.zeros_like(true_positive), where=denominator > 0,
            ))
        macro_f1_replicates[key] = np.mean(np.stack(class_f1, axis=1), axis=1)
        observed_class_f1 = []
        for label in classes:
            true_indices = class_indices[label]
            tp = np.sum(pred[true_indices] == label)
            predicted_count = np.sum(pred == label)
            observed_class_f1.append(2 * tp / (len(true_indices) + predicted_count)
                                     if len(true_indices) + predicted_count else 0.0)
        observed_macro_f1[key] = float(np.mean(observed_class_f1))

    pair_results = []
    raw_p = []
    for model_a, model_b in itertools.combinations(model_keys, 2):
        a_correct = predictions[model_a] == labels
        b_correct = predictions[model_b] == labels
        a_only = int(np.sum(a_correct & ~b_correct))
        b_only = int(np.sum(~a_correct & b_correct))
        discordant = a_only + b_only
        p_value = float(binomtest(a_only, discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
        accuracy_difference = observed_accuracy[model_a] - observed_accuracy[model_b]
        accuracy_bootstrap_difference = accuracy_replicates[model_a] - accuracy_replicates[model_b]
        f1_difference = observed_macro_f1[model_a] - observed_macro_f1[model_b]
        f1_bootstrap_difference = macro_f1_replicates[model_a] - macro_f1_replicates[model_b]
        pair_results.append({
            "model_a": model_a,
            "model_b": model_b,
            "sample_count": n,
            "accuracy_difference_a_minus_b": accuracy_difference,
            "accuracy_difference_paired_stratified_bootstrap_95_ci": [
                float(v) for v in np.quantile(accuracy_bootstrap_difference, [0.025, 0.975])
            ],
            "macro_f1_difference_a_minus_b": f1_difference,
            "macro_f1_difference_paired_stratified_bootstrap_95_ci": [
                float(v) for v in np.quantile(f1_bootstrap_difference, [0.025, 0.975])
            ],
            "mcnemar_exact": {
                "a_correct_b_wrong": a_only,
                "a_wrong_b_correct": b_only,
                "discordant_pairs": discordant,
                "p_value_two_sided": p_value,
            },
        })
        raw_p.append(p_value)

    # Holm family-wise adjustment for the six exploratory paired accuracy tests.
    order = np.argsort(raw_p)
    adjusted = [None] * len(raw_p)
    running_max = 0.0
    m = len(raw_p)
    for rank, index in enumerate(order):
        running_max = max(running_max, min(1.0, (m - rank) * raw_p[index]))
        adjusted[index] = running_max
    for result, p_adjusted in zip(pair_results, adjusted):
        result["mcnemar_exact"]["p_value_holm_adjusted"] = float(p_adjusted)
    return {
        "sample_count": n,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": seed,
        "bootstrap_method": "Image-paired percentile bootstrap, stratified within observed true classes; each replicate preserves class supports and draws the same image indices for every model.",
        "mcnemar_method": "Exact two-sided McNemar binomial test on discordant paired correctness outcomes; Holm adjustment across the six exploratory pairwise accuracy tests.",
        "interpretation": "Descriptive comparisons on one small external sample. Bootstrap intervals and tests do not address dataset selection, mapping uncertainty, or generalization to a target population.",
        "comparisons": pair_results,
    }


def plot_confidence(grouped, output, dataset_display):
    keys = list(grouped)
    values, colors = [], []
    short_names = {
        "mobilenet": "MobileNetV3",
        "mobilenet_clahe": "MobileNetV3 + CLAHE",
        "deit_tiny": "DeiT-Tiny",
        "hybrid": "Hybrid",
    }
    positions, tick_labels, group_centers, group_labels = [], [], [], []
    for model_index, model in enumerate(keys):
        pair_positions = [model_index * 3 + 1, model_index * 3 + 2]
        group_centers.append(np.mean(pair_positions))
        group_labels.append(short_names.get(model, model))
        for outcome, color in (("correct", "#238b45"), ("incorrect", "#cb181d")):
            vals = [r["confidence"] for r in grouped[model] if r["correct"] == (outcome == "correct")]
            values.append(vals if vals else [np.nan])
            colors.append(color)
            positions.append(pair_positions[0] if outcome == "correct" else pair_positions[1])
            tick_labels.append(f"{outcome.title()}\n(n={len(vals)})")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    bp = ax.boxplot(values, positions=positions, patch_artist=True, showfliers=False, widths=0.62)
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.68)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Maximum softmax probability")
    ax.set_xticks(positions, tick_labels, fontsize=8)
    for center, label in zip(group_centers, group_labels):
        ax.text(center, -0.20, label, ha="center", va="top", fontsize=9,
                transform=ax.get_xaxis_transform())
    ax.set_xlim(0.4, positions[-1] + 0.6)
    ax.set_title(f"{dataset_display} max-softmax score by correctness")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(handles=[
        Patch(facecolor="#238b45", alpha=0.68, label="Correct"),
        Patch(facecolor="#cb181d", alpha=0.68, label="Incorrect"),
    ], loc="lower left", frameon=False)
    fig.subplots_adjust(bottom=0.28, left=0.08, right=0.99, top=0.9)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_f1_degradation(analyses, output, dataset_display):
    model_names = list(analyses)
    all_classes = list(next(iter(analyses.values()))["per_class_plantvillage_to_external"])
    classes = [name.replace("___", "\n") for name in all_classes]
    x = np.arange(len(all_classes))
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharey=True)
    model_colors = {"mobilenet": "#2171b5", "mobilenet_clahe": "#cb181d",
                    "deit_tiny": "#238b45", "hybrid": "#756bb1"}
    for idx, (ax, model) in enumerate(zip(axes.flat, model_names)):
        details = analyses[model]["per_class_plantvillage_to_external"]
        pv = [details[name]["plantvillage_f1"] for name in all_classes]
        ext = [details[name]["external_f1"] for name in all_classes]
        width = 0.38
        color = model_colors.get(model, "#2171b5")
        ax.bar(x - width / 2, pv, width, color=color, alpha=0.42, label="PlantVillage")
        ax.bar(x + width / 2, ext, width, color=color, alpha=0.95, label=dataset_display)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x, classes, fontsize=7)
        ax.set_title(analyses[model]["model_name"])
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Class F1")
    axes[1, 0].set_ylabel("Class F1")
    axes[0, 0].legend(ncols=2, fontsize=8, frameon=False)
    fig.suptitle(f"Per-class F1: PlantVillage test vs {dataset_display}", y=0.995)
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "results/domain_shift/plantdoc_predictions.csv")
    parser.add_argument("--external-results", type=Path, default=ROOT / "results/domain_shift")
    parser.add_argument("--external-slug", default="plantdoc")
    parser.add_argument("--pv-results", type=Path, default=ROOT / "results/final_results.json")
    parser.add_argument("--output-json", type=Path, default=ROOT / "results/domain_shift/confidence_analysis.json")
    parser.add_argument("--plot-dir", type=Path, default=ROOT / "results/plots/domain_shift")
    parser.add_argument("--high-confidence-threshold", type=float, default=0.9)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260924)
    args = parser.parse_args()
    if not 0 < args.high_confidence_threshold <= 1 or args.top_n < 1 or args.bootstrap_replicates < 100:
        raise ValueError("Threshold must be in (0, 1], top-n positive, and bootstrap replicates at least 100")

    rows = load_predictions(args.predictions)
    if not rows:
        raise RuntimeError(f"No predictions in {args.predictions}")
    final = json.loads(args.pv_results.read_text(encoding="utf-8"))
    grouped = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)
    analyses = {}
    for model in grouped:
        external_path = args.external_results / f"{model}_{args.external_slug}_test_results.json"
        external = json.loads(external_path.read_text(encoding="utf-8"))
        pv = final["plantvillage_full_results"][model]
        analyses[model] = analyze_model(
            model, grouped[model], external, pv,
            args.high_confidence_threshold, args.top_n,
        )
    output = {
        "external_dataset": args.external_slug,
        "confidence_definition": "Maximum softmax probability of the top-1 output; not a calibrated probability of correctness.",
        "high_confidence_threshold": args.high_confidence_threshold,
        "method_note": "Descriptive finite-sample summaries only. No calibration metric or calibrated-uncertainty claim is made.",
        "models": analyses,
        "paired_model_comparisons": paired_comparisons(
            grouped, args.bootstrap_replicates, args.bootstrap_seed
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    dataset_display = {
        "plantdoc": "PlantDoc",
        "plantseg_v7": "PlantSeg v7",
    }.get(args.external_slug, args.external_slug.replace("_", " ").title())
    plot_confidence(grouped, args.plot_dir / f"{args.external_slug}_confidence_correct_vs_incorrect.png", dataset_display)
    plot_f1_degradation(analyses, args.plot_dir / f"{args.external_slug}_per_class_f1_degradation.png", dataset_display)
    print(f"Wrote {args.output_json}")
    print(f"Analyzed {len(rows)} predictions across {len(grouped)} models; confidence threshold={args.high_confidence_threshold:.2f}.")


if __name__ == "__main__":
    main()
