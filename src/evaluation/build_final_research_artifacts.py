"""Consolidate audited results into final tables and publication figures."""
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from src.inference.common import CLASS_NAMES, MODEL_SPECS, ROOT

OUT = ROOT / "results/plots/final"
METRICS_DIR = ROOT / "results/metrics"
DATASETS = {
    "plantdoc": {"name": "PlantDoc", "path": ROOT / "results/domain_shift"},
    "plantseg_v7": {"name": "PlantSeg v7", "path": ROOT / "results/domain_shift/plantseg_v7"},
}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260924


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stratified_draws(matrix, mapped_names, rng, replicates=BOOTSTRAP_REPLICATES):
    matrix = np.asarray(matrix, dtype=np.int64)
    indices = [CLASS_NAMES.index(name) for name in mapped_names]
    rows = matrix[indices, :]
    supports = rows.sum(axis=1)
    if np.any(supports <= 0):
        raise ValueError("Every mapped class must have nonzero support")
    draws = np.stack([
        rng.multinomial(int(n), row / n, size=replicates)
        for row, n in zip(rows, supports)
    ], axis=1)
    tp = np.stack([draws[:, i, class_index] for i, class_index in enumerate(indices)], axis=1)
    predicted = draws.sum(axis=1)
    pred_mapped = predicted[:, indices]
    class_f1 = 2 * tp / np.maximum(supports[None, :] + pred_mapped, 1)
    accuracy = tp.sum(axis=1) / supports.sum()
    macro_f1 = class_f1.mean(axis=1)
    return accuracy, macro_f1


def ci(values):
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def pv_same_class_metrics(matrix, mapped_names):
    matrix = np.asarray(matrix, dtype=np.int64)
    indices = [CLASS_NAMES.index(name) for name in mapped_names]
    rows = matrix[indices, :]
    n = int(rows.sum())
    tp = np.asarray([matrix[index, index] for index in indices])
    supports = rows.sum(axis=1)
    pred_counts = matrix[indices, :].sum(axis=0)
    precision = tp / np.maximum(pred_counts[indices], 1)
    recall = tp / supports
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    accuracy = float(tp.sum() / n)
    return {
        "sample_count": n,
        "mapped_classes": len(mapped_names),
        "accuracy": accuracy,
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "f1_weighted": float(np.average(f1, weights=supports)),
    }


def format_pct(value, digits=2):
    return f"{100 * value:.{digits}f}%"


def create_system_diagram():
    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("LeafGuard cross-domain evaluation system", fontsize=17, pad=18)

    def box(x, y, w, h, text, color="#e8f1fb", fontsize=10, edge="#355c7d"):
        shape = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                               facecolor=color, edgecolor=edge, linewidth=1.5)
        ax.add_patch(shape)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.5, color="#485563"))

    box(0.3, 5.1, 2.5, 1.3, "Input image\n224 × 224", color="#edf2f7")
    box(0.3, 2.6, 2.5, 1.4, "ImageNet normalization\nCLAHE only for CLAHE model", color="#edf2f7")
    arrow(1.55, 5.1, 1.55, 4.0)
    branches = [
        (3.5, 6.0, "MobileNetV3-Small\n576 features"),
        (3.5, 4.35, "MobileNetV3-Small + CLAHE\n576 features"),
        (3.5, 2.7, "DeiT-Tiny\n192 features"),
        (3.5, 1.05, "Hybrid: CNN 576 + DeiT 192\nconcat 768 → 512 → 256 → 10"),
    ]
    for x, y, text in branches:
        box(x, y, 3.5, 1.15, text, color="#e8f5e9" if "Hybrid" in text else "#e8f1fb", fontsize=9.5)
        arrow(2.8, 3.3, x, y + 0.58)
    box(7.8, 3.4, 2.3, 1.4, "Top-1 class\nmax-softmax score\n10 output labels", color="#fff4df")
    for _, y, _ in branches:
        arrow(7.0, y + 0.58, 7.8, 4.1)
    box(11.0, 5.25, 2.6, 1.35, "PlantVillage\n2,054 test images\n10 classes", color="#f2eafb")
    box(11.0, 3.25, 2.6, 1.35, "PlantDoc\n83 mapped images\n9 classes", color="#f2eafb")
    box(11.0, 1.25, 2.6, 1.35, "PlantSeg v7\n143 mapped images\n7 classes", color="#f2eafb")
    for y in (5.925, 3.925, 1.925):
        arrow(10.1, 4.1, 11.0, y)
    ax.text(7.7, 0.45, "Frozen checkpoints • identical samples per model within each dataset • no retraining",
            fontsize=10, color="#374151")
    fig.tight_layout()
    fig.savefig(OUT / "leafguard_evaluation_system.png", dpi=260, bbox_inches="tight")
    fig.savefig(OUT / "leafguard_evaluation_system.svg", bbox_inches="tight")
    plt.close(fig)


def plot_paired_metric(model_order, records, metric, output_name, ylabel, title):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=True)
    colors = {"mobilenet": "#2171b5", "mobilenet_clahe": "#cb181d",
              "deit_tiny": "#238b45", "hybrid": "#756bb1"}
    labels = ["MobileNetV3", "MobileNetV3 + CLAHE", "DeiT-Tiny", "Hybrid"]
    x = np.arange(len(model_order))
    width = 0.36
    for ax, slug in zip(axes, DATASETS):
        pv_values = [records[key]["matched_plantvillage"][slug][metric] for key in model_order]
        ext_values = [records[key]["external"][slug]["metrics"][metric] for key in model_order]
        ax.bar(x - width / 2, pv_values, width, color=[colors[key] for key in model_order],
               alpha=0.42, label="PlantVillage (same classes)")
        ax.bar(x + width / 2, ext_values, width, color=[colors[key] for key in model_order],
               alpha=0.95, label=DATASETS[slug]["name"])
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_ylim(0, 1.03)
        ax.set_ylabel(ylabel)
        counts = [records[model_order[0]]["external"][slug]["metrics"]["total"]]
        ax.set_title(f"{DATASETS[slug]['name']} (n={counts[0]})")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / output_name, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_domain_gaps(model_order, records, metric, output_name, title):
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(len(model_order))
    width = 0.36
    names = ["MobileNetV3", "MobileNetV3 + CLAHE", "DeiT-Tiny", "Hybrid"]
    doc = [100 * records[key]["domain_gap"]["plantdoc"][metric] for key in model_order]
    seg = [100 * records[key]["domain_gap"]["plantseg_v7"][metric] for key in model_order]
    ax.bar(x - width / 2, doc, width, color="#2171b5", label="PlantDoc")
    ax.bar(x + width / 2, seg, width, color="#756bb1", label="PlantSeg v7")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_ylabel("PlantVillage same-class metric − external metric (percentage points)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / output_name, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_confusions(dataset_slug, results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    labels = ["Apple scab", "Apple black rot", "Corn common rust", "Grape black rot",
              "Grape healthy", "Potato early blight", "Potato late blight",
              "Tomato early blight", "Tomato late blight", "Tomato healthy"]
    for ax, model_key in zip(axes.flat, MODEL_SPECS):
        result = results[model_key]["external"][dataset_slug]
        cm = np.asarray(result["confusion_matrix"])
        image = ax.imshow(cm, cmap="Blues", vmin=0)
        ax.set_xticks(range(10), labels, rotation=55, ha="right", fontsize=7)
        ax.set_yticks(range(10), labels, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(result["model"])
        threshold = cm.max() / 2 if cm.max() else 0
        for row, col in zip(*np.nonzero(cm)):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", fontsize=7,
                    color="white" if cm[row, col] > threshold else "black")
    fig.suptitle(f"{DATASETS[dataset_slug]['name']} confusion matrices", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / f"{dataset_slug}_confusion_matrices.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT / "results/final_results.json").read_text(encoding="utf-8"))
    pv_results = {
        key: json.loads((ROOT / "results/audit" / f"{key}_plantvillage_verified.json").read_text(encoding="utf-8"))
        for key in MODEL_SPECS
    }
    ext_results, confidence, intervals = {}, {}, {}
    for slug, dataset in DATASETS.items():
        ext_results[slug] = {
            key: json.loads((dataset["path"] / f"{key}_{slug}_test_results.json").read_text(encoding="utf-8"))
            for key in MODEL_SPECS
        }
        confidence[slug] = json.loads((dataset["path"] / "confidence_analysis.json").read_text(encoding="utf-8"))
        intervals[slug] = json.loads((dataset["path"] / f"{slug}_stratified_intervals.json").read_text(encoding="utf-8"))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    records, master_rows, failure_rows, per_class_rows, paired_rows, confidence_rows = {}, [], [], [], [], []
    for model_key, spec in MODEL_SPECS.items():
        pv_cm = pv_results[model_key]["confusion_matrix"]
        record = {
            "model": spec["name"],
            "model_key": model_key,
            "plantvillage_full": {
                **pv_results[model_key]["metrics"],
                "sample_count": pv_results[model_key]["metrics"]["total"],
                "mapped_classes": 10,
            },
            "matched_plantvillage": {},
            "external": {},
            "domain_gap": {},
            "resources": base["resource_measurements"][model_key],
            "confidence": {},
        }
        pv_full_acc_reps, pv_full_f1_reps = stratified_draws(pv_cm, CLASS_NAMES, rng)
        record["plantvillage_full"]["accuracy_95_ci"] = ci(pv_full_acc_reps)
        record["plantvillage_full"]["macro_f1_95_ci"] = ci(pv_full_f1_reps)
        master = {
            "model": spec["name"],
            "model_key": model_key,
            "plantvillage_n": pv_results[model_key]["metrics"]["total"],
            "plantvillage_accuracy": pv_results[model_key]["metrics"]["accuracy"],
            "plantvillage_macro_f1": pv_results[model_key]["metrics"]["f1_macro"],
            "plantvillage_accuracy_95_ci_low": record["plantvillage_full"]["accuracy_95_ci"][0],
            "plantvillage_accuracy_95_ci_high": record["plantvillage_full"]["accuracy_95_ci"][1],
            "plantvillage_macro_f1_95_ci_low": record["plantvillage_full"]["macro_f1_95_ci"][0],
            "plantvillage_macro_f1_95_ci_high": record["plantvillage_full"]["macro_f1_95_ci"][1],
            "parameters": record["resources"]["parameter_count"],
            "checkpoint_mib": record["resources"]["checkpoint_size_mib"],
            "latency_ms_batch1": record["resources"]["inference_latency_ms_batch1"],
            "approximate_flops_batch1": record["resources"]["approximate_flops_batch1"],
        }
        for slug, dataset in DATASETS.items():
            result = ext_results[slug][model_key]
            mapped_names = result["mapped_classes"]
            matched_pv = pv_same_class_metrics(pv_cm, mapped_names)
            pv_acc_rep, pv_f1_rep = stratified_draws(pv_cm, mapped_names, rng)
            external_acc_rep, external_f1_rep = stratified_draws(result["confusion_matrix"], mapped_names, rng)
            external_ci = {
                "accuracy_95_ci": ci(external_acc_rep),
                "macro_f1_95_ci": ci(external_f1_rep),
                "resamples": BOOTSTRAP_REPLICATES,
                "method": "Stratified image bootstrap from confusion-matrix counts; outcomes resampled within each observed true class, class supports fixed.",
            }
            gap_acc_rep = pv_acc_rep - external_acc_rep
            gap_f1_rep = pv_f1_rep - external_f1_rep
            gap = {
                "accuracy": matched_pv["accuracy"] - result["metrics"]["accuracy"],
                "macro_f1": matched_pv["f1_macro"] - result["metrics"]["f1_macro"],
                "accuracy_95_ci": ci(gap_acc_rep),
                "macro_f1_95_ci": ci(gap_f1_rep),
                "bootstrap_method": "Independent stratified image bootstrap in PlantVillage and external domains; conditional on each domain's observed class supports.",
            }
            record["matched_plantvillage"][slug] = matched_pv | {
                "accuracy_95_ci": ci(pv_acc_rep),
                "macro_f1_95_ci": ci(pv_f1_rep),
            }
            record["external"][slug] = result | {"uncertainty": external_ci}
            record["domain_gap"][slug] = gap
            confidence_model = confidence[slug]["models"][model_key]
            record["confidence"][slug] = confidence_model
            for outcome, summary in confidence_model["confidence_by_outcome"].items():
                confidence_rows.append({
                    "model": spec["name"], "model_key": model_key,
                    "dataset": dataset["name"], "outcome": outcome,
                    "count": summary["count"], "mean": summary["mean"], "median": summary["median"],
                    **{f"{quantile}_confidence": value for quantile, value in summary["quantiles"].items()},
                    "score_definition": "maximum softmax; not calibrated uncertainty",
                })
            master[f"{slug}_n"] = result["metrics"]["total"]
            master[f"{slug}_accuracy"] = result["metrics"]["accuracy"]
            master[f"{slug}_macro_f1"] = result["metrics"]["f1_macro"]
            master[f"{slug}_accuracy_95_ci_low"] = external_ci["accuracy_95_ci"][0]
            master[f"{slug}_accuracy_95_ci_high"] = external_ci["accuracy_95_ci"][1]
            master[f"{slug}_macro_f1_95_ci_low"] = external_ci["macro_f1_95_ci"][0]
            master[f"{slug}_macro_f1_95_ci_high"] = external_ci["macro_f1_95_ci"][1]
            master[f"{slug}_matched_pv_accuracy"] = matched_pv["accuracy"]
            master[f"{slug}_matched_pv_macro_f1"] = matched_pv["f1_macro"]
            master[f"{slug}_accuracy_gap"] = gap["accuracy"]
            master[f"{slug}_macro_f1_gap"] = gap["macro_f1"]
            master[f"{slug}_accuracy_gap_95_ci_low"] = gap["accuracy_95_ci"][0]
            master[f"{slug}_accuracy_gap_95_ci_high"] = gap["accuracy_95_ci"][1]
            master[f"{slug}_macro_f1_gap_95_ci_low"] = gap["macro_f1_95_ci"][0]
            master[f"{slug}_macro_f1_gap_95_ci_high"] = gap["macro_f1_95_ci"][1]

            for label, scores in result["per_class"].items():
                pv_scores = pv_results[model_key]["per_class"][label]
                per_class_rows.append({
                    "model": spec["name"], "model_key": model_key,
                    "dataset": dataset["name"], "class": label,
                    "support": scores["support"],
                    "precision": scores["precision"], "recall": scores["recall"],
                    "f1": scores["f1-score"],
                    "plantvillage_f1_same_class": pv_scores["f1-score"],
                    "f1_external_minus_plantvillage": scores["f1-score"] - pv_scores["f1-score"],
                })
            worst = confidence_model["worst_external_class_by_f1"]
            confusion = confidence_model["most_common_confusion_pairs"]
            high = confidence_model["high_confidence_error_summary"]
            failure_rows.append({
                "model": spec["name"], "model_key": model_key,
                "dataset": dataset["name"], "sample_count": result["metrics"]["total"],
                "worst_class": worst["label"], "worst_class_f1": worst["f1"],
                "worst_class_support": worst["support"],
                "most_common_confusion": (
                    f"{confusion[0]['true_label']} → {confusion[0]['predicted_label']} (n={confusion[0]['count']})"
                    if confusion else "none"
                ),
                "high_confidence_threshold": high["threshold"],
                "high_confidence_predictions": high["predictions_at_or_above_threshold"],
                "high_confidence_errors": high["incorrect_at_or_above_threshold"],
                "high_confidence_error_fraction": high["error_fraction_among_predictions_at_or_above_threshold"],
                "macro_f1": result["metrics"]["f1_macro"],
            })
            pair_data = confidence[slug]["paired_model_comparisons"]
            for comparison in pair_data["comparisons"]:
                paired_rows.append({
                    "dataset": dataset["name"], **comparison,
                })
        records[model_key] = record
        master_rows.append(master)

    output = {
        "project": "LeafGuard",
        "generated_from": [
            "results/final_results.json",
            "results/audit/*_plantvillage_verified.json",
            "results/domain_shift/*_test_results.json",
            "results/domain_shift/*/confidence_analysis.json",
            "results/audit/plantseg_v7_evaluation_audit.json",
        ],
        "plantseg_evaluation_audit_status": json.loads((ROOT / "results/audit/plantseg_v7_evaluation_audit.json").read_text(encoding="utf-8"))["audit_status"],
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "method": "Stratified image bootstrap from saved confusion matrices; separate independent domain resampling for domain-gap intervals; observed class supports are fixed.",
            "interpretation_limit": "Intervals are conditional on observed dataset samples, mappings, source and supports; they do not capture dataset selection or source-population uncertainty.",
        },
        "models": records,
        "paired_comparisons": {slug: confidence[slug]["paired_model_comparisons"] for slug in DATASETS},
        "limitations": [
            "PlantDoc n=83 across nine mapped classes; PlantSeg n=143 across seven mapped classes, with potato early blight n=9.",
            "External samples are web-sourced curated subsets, not probability samples of farms.",
            "PlantSeg's observed images include mixed plant organs and visual graphics; no causal subgroup inference is made.",
            "Maximum softmax scores are not calibrated uncertainty.",
            "No new model training or robustness intervention was performed; external test results were kept as evaluation evidence.",
        ],
    }
    (ROOT / "results/final_research_results.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(METRICS_DIR / "master_results.csv", master_rows)
    write_csv(METRICS_DIR / "failure_analysis.csv", failure_rows)
    write_csv(METRICS_DIR / "per_class_external_and_matched_pv.csv", per_class_rows)
    write_csv(METRICS_DIR / "paired_model_comparisons.csv", paired_rows)
    write_csv(METRICS_DIR / "confidence_by_correctness.csv", confidence_rows)

    master_markdown = [
        "# Master results across controlled and external test sets", "",
        "External metrics average over each dataset's mapped classes. Domain gaps compare each external result with PlantVillage restricted to the same true classes. All entries are from frozen checkpoints; no retraining was performed.", "",
        "| Model | PV accuracy (10) | PV macro F1 (10) | PlantDoc accuracy / F1 (9) | PlantSeg accuracy / F1 (7) | PlantDoc acc / F1 gap | PlantSeg acc / F1 gap | Parameters | Latency (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in master_rows:
        master_markdown.append(
            f"| {row['model']} | {format_pct(row['plantvillage_accuracy'])} | {format_pct(row['plantvillage_macro_f1'])} | "
            f"{format_pct(row['plantdoc_accuracy'])} / {format_pct(row['plantdoc_macro_f1'])} | "
            f"{format_pct(row['plantseg_v7_accuracy'])} / {format_pct(row['plantseg_v7_macro_f1'])} | "
            f"{100*row['plantdoc_accuracy_gap']:.2f} / {100*row['plantdoc_macro_f1_gap']:.2f} pp | "
            f"{100*row['plantseg_v7_accuracy_gap']:.2f} / {100*row['plantseg_v7_macro_f1_gap']:.2f} pp | "
            f"{row['parameters']:,} | {row['latency_ms_batch1']:.2f} |"
        )
    master_markdown.extend([
        "",
        "Accuracy and macro-F1 95% bootstrap intervals are in `results/metrics/master_results.csv` as low/high columns; domain-gap intervals are also recorded there. The bootstrap conditions on each dataset's observed class supports and does not address source selection.",
        "",
        "Point estimates alone do not establish a ranking. Paired bootstrap intervals and exact McNemar tests for the same external images are in `results/metrics/paired_model_comparisons.csv` and the full JSON artifacts.",
        "",
        "PlantVillage values in the first two columns use all ten classes. The same-class PlantVillage baselines used to calculate each domain gap are kept separately in the CSV and JSON. PlantDoc and PlantSeg overlap on different classes and have different sample supports, so their external scores should not be compared as if they were the same test distribution.",
        "",
    ])
    (ROOT / "research/master_results.md").write_text("\n".join(master_markdown), encoding="utf-8")

    failure_lines = [
        "# External failure summary", "",
        "High-confidence error fraction is the fraction incorrect among predictions with max-softmax ≥0.90; it is descriptive and is not a calibration metric.", "",
        "| Model | Dataset | n | Worst class (F1; n) | Most common confusion | ≥0.90 predictions / errors | High-confidence error fraction | Macro F1 |",
        "|---|---|---:|---|---|---:|---:|---:|",
    ]
    for row in failure_rows:
        failure_lines.append(
            f"| {row['model']} | {row['dataset']} | {row['sample_count']} | {row['worst_class']} ({row['worst_class_f1']:.3f}; {row['worst_class_support']}) | {row['most_common_confusion']} | {row['high_confidence_predictions']} / {row['high_confidence_errors']} | {row['high_confidence_error_fraction']:.1%} | {row['macro_f1']:.3f} |"
        )
    failure_lines.append("")
    (ROOT / "research/failure_analysis.md").write_text("\n".join(failure_lines), encoding="utf-8")
    robustness_rows = []
    for row in master_rows:
        for slug, dataset in DATASETS.items():
            robustness_rows.append({
                "experiment": f"{row['model']} on {dataset['name']}",
                "in_domain_accuracy_same_classes": row[f"{slug}_matched_pv_accuracy"],
                "in_domain_macro_f1_same_classes": row[f"{slug}_matched_pv_macro_f1"],
                "external_accuracy": row[f"{slug}_accuracy"],
                "external_macro_f1": row[f"{slug}_macro_f1"],
                "accuracy_gap": row[f"{slug}_accuracy_gap"],
                "macro_f1_gap": row[f"{slug}_macro_f1_gap"],
                "interpretation": "Frozen-checkpoint external evaluation only; descriptive domain gap, no causal robustness claim.",
            })
    write_csv(METRICS_DIR / "robustness_comparison.csv", robustness_rows)
    robustness_lines = [
        "# Robustness comparison", "",
        "| Frozen model / external set | PV accuracy (same classes) | External accuracy | Accuracy gap | PV macro F1 (same classes) | External macro F1 | Macro-F1 gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in robustness_rows:
        robustness_lines.append(
            f"| {row['experiment']} | {format_pct(row['in_domain_accuracy_same_classes'])} | "
            f"{format_pct(row['external_accuracy'])} | {100*row['accuracy_gap']:.2f} pp | "
            f"{format_pct(row['in_domain_macro_f1_same_classes'])} | {format_pct(row['external_macro_f1'])} | "
            f"{100*row['macro_f1_gap']:.2f} pp |"
        )
    robustness_lines.extend([
        "",
        "These are frozen-checkpoint comparisons, not intervention results. The PlantVillage baseline is restricted to the same mapped classes as each external test. Gaps are descriptive; independent stratified-bootstrap intervals are available in `results/metrics/master_results.csv`.",
        "",
        "No controlled augmentation or training intervention was performed because both available external datasets are now evaluation evidence. Using their observed test failures to select a training modification would convert them into development data and leave no independent external evaluation set.",
        "",
    ])
    (ROOT / "research/robustness_comparison.md").write_text("\n".join(robustness_lines), encoding="utf-8")

    model_order = list(MODEL_SPECS)
    plot_paired_metric(model_order, records, "accuracy", "plantvillage_vs_external_accuracy.png",
                       "Accuracy", "PlantVillage same-class vs external accuracy")
    plot_paired_metric(model_order, records, "f1_macro", "plantvillage_vs_external_macro_f1.png",
                       "Macro F1", "PlantVillage same-class vs external macro F1")
    plot_domain_gaps(model_order, records, "accuracy", "domain_gap_accuracy.png", "Same-class accuracy gap")
    plot_domain_gaps(model_order, records, "macro_f1", "domain_gap_macro_f1.png", "Same-class macro-F1 gap")
    plot_confusions("plantdoc", records)
    plot_confusions("plantseg_v7", records)
    create_system_diagram()

    resource = base["resource_measurements"]
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    for key, spec in MODEL_SPECS.items():
        point = resource[key]
        x_value = point["parameter_count"] / 1e6
        y_value = point["inference_latency_ms_batch1"]
        ax.scatter(x_value, y_value, s=90,
                   label=spec["name"])
        label = spec["name"].replace("MobileNetV3-Small", "MobileNetV3")
        if key == "mobilenet":
            offset, align = (8, -14), "left"
        elif key == "mobilenet_clahe":
            offset, align = (8, 9), "left"
        elif key == "hybrid":
            offset, align = (-8, 1), "right"
        else:
            offset, align = (8, 8), "left"
        ax.annotate(label, (x_value, y_value), xytext=offset, textcoords="offset points",
                    ha=align, fontsize=8)
    ax.set_xlabel("Trainable parameters (millions)")
    ax.set_ylabel("Batch-one forward latency (ms/image)")
    ax.set_xlim(0.8, 8.2)
    ax.set_ylim(5.5, 17.0)
    ax.set_title("Model size and measured inference latency")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "parameter_count_vs_latency.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    for slug, dataset in DATASETS.items():
        if slug == "plantdoc":
            confidence_figure = ROOT / "results/plots/domain_shift/plantdoc_confidence_correct_vs_incorrect.png"
        else:
            confidence_figure = ROOT / "results/plots/domain_shift" / slug / f"{slug}_confidence_correct_vs_incorrect.png"
        target = OUT / f"{slug}_confidence_correct_vs_incorrect.png"
        if confidence_figure.is_file():
            shutil.copy2(confidence_figure, target)
    gradcam_source = ROOT / "results/explainability/paired_examples/selected_gradcam_examples.png"
    if gradcam_source.is_file():
        shutil.copy2(gradcam_source, OUT / "selected_gradcam_examples.png")
    print(f"Wrote final results, tables, and figures under {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
