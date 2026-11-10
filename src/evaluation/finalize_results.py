"""Re-verify fixed PlantVillage checkpoints and build research comparison artifacts."""
import json
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.inference.common import CLASS_NAMES, MODEL_SPECS, ROOT, build_model, make_transform
from src.security.integrity import verify_checkpoint

PV_SOURCE_FILES = {
    "mobilenet": ROOT / "results/metrics/mobilenet_test_results.json",
    "mobilenet_clahe": ROOT / "results/metrics/mobilenet_clahe_test_results.json",
    "deit_tiny": ROOT / "results/metrics/deit_tiny_test_results.json",
    "hybrid": ROOT / "results/metrics/hybrid_test_results.json",
}
FRIENDLY = ["MobileNetV3", "MobileNetV3 + CLAHE", "DeiT-Tiny", "Hybrid"]


def read_metrics(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("metrics", data)
    return {
        "accuracy": values.get("accuracy"),
        "precision_macro": values.get("precision_macro", values.get("macro_precision")),
        "recall_macro": values.get("recall_macro", values.get("macro_recall")),
        "f1_macro": values.get("f1_macro", values.get("macro_f1")),
        "correct": values.get("correct", values.get("correct_predictions")),
        "total": values.get("total", values.get("test_samples")),
    }


def metrics_for_mapped_subset(confusion, all_class_names, mapped_class_names):
    """Compute PV metrics on the same ground-truth classes used by PlantDoc."""
    cm = np.asarray(confusion, dtype=np.int64)
    ids = [all_class_names.index(name) for name in mapped_class_names]
    subset = cm[ids, :]
    true_support = subset.sum(axis=1)
    predicted_support = subset[:, ids].sum(axis=0)
    true_positive = np.asarray([cm[class_id, class_id] for class_id in ids])
    precision = np.divide(true_positive, predicted_support,
                          out=np.zeros_like(true_positive, dtype=float), where=predicted_support > 0)
    recall = np.divide(true_positive, true_support,
                       out=np.zeros_like(true_positive, dtype=float), where=true_support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    total = int(true_support.sum())
    return {
        "class_count": len(ids),
        "class_names": mapped_class_names,
        "total": total,
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "per_class": {
            name: {
                "precision": float(p), "recall": float(r), "f1-score": float(f),
                "support": int(s),
            }
            for name, p, r, f, s in zip(mapped_class_names, precision, recall, f1, true_support)
        },
    }


def verify_plantvillage(model_key, device, batch_size=32):
    spec = MODEL_SPECS[model_key]
    dataset = ImageFolder(ROOT / "data/processed/test", transform=make_transform(spec["clahe"]))
    if dataset.classes != CLASS_NAMES:
        raise RuntimeError(f"Test class order mismatch: {dataset.classes}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    model = build_model(model_key, device=device)
    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())
    report = classification_report(y_true, y_pred, labels=list(range(10)),
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    result = {
        "model": spec["name"],
        "checkpoint": str(spec["checkpoint"].relative_to(ROOT)),
        "dataset": "PlantVillage held-out test split",
        "preprocessing": {"image_size": [224, 224], "clahe": spec["clahe"],
                          "clahe_clip_limit": 2.0 if spec["clahe"] else None,
                          "clahe_tile_grid_size": [8, 8] if spec["clahe"] else None},
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "correct": int(sum(a == b for a, b in zip(y_true, y_pred))),
            "total": len(y_true),
        },
        "per_class": {name: report[name] for name in CLASS_NAMES},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(10))).tolist(),
        "confusion_matrix_labels": CLASS_NAMES,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def measure_resources(model_key, device, latency_iterations=60):
    spec = MODEL_SPECS[model_key]
    model = build_model(model_key, device=device)
    parameter_count = sum(p.numel() for p in model.parameters())
    size_bytes = spec["checkpoint"].stat().st_size
    x = torch.zeros(1, 3, 224, 224, device=device)
    latency_trials = []
    with torch.inference_mode():
        for _ in range(8):
            model(x)
        for _ in range(3):
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start = time.perf_counter()
            for _ in range(latency_iterations):
                model(x)
            if device.type == "cuda":
                end_event.record()
                end_event.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                elapsed_ms = (time.perf_counter() - start) * 1000
            latency_trials.append(elapsed_ms / latency_iterations)
    latency_ms = statistics.median(latency_trials)
    latency_cv = statistics.pstdev(latency_trials) / statistics.mean(latency_trials)

    # Torch's built-in FLOP instrumentation currently covers common Conv/MatMul ops;
    # report the value as an estimate because some operations are not counted.
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    try:
        with profile(activities=activities, record_shapes=True, with_flops=True) as prof:
            with torch.inference_mode():
                model(x)
        flops = sum(int(event.flops or 0) for event in prof.key_averages())
    except Exception as exc:
        flops = None
        flops_note = f"Profiler did not provide a FLOP count: {type(exc).__name__}: {exc}"
    else:
        flops_note = "Profiler-based approximate count; unsupported operators are omitted."
        if flops == 0:
            flops = None
            flops_note = "Profiler provided no supported FLOPs for this model."
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "parameter_count": int(parameter_count),
        "checkpoint_size_bytes": int(size_bytes),
        "checkpoint_size_mib": round(size_bytes / (1024 ** 2), 3),
        "inference_latency_ms_batch1": float(latency_ms),
        "latency_trials_ms_batch1": [float(x) for x in latency_trials],
        "latency_trial_cv": float(latency_cv),
        "latency_method": "Median of three 60-forward batches after eight warmups; CUDA events on GPU or perf_counter on CPU; model forward only, excluding decode and image preprocessing.",
        "latency_device": str(device),
        "approximate_flops_batch1": flops,
        "flops_note": flops_note,
    }


def draw_bar(values, title, ylabel, path, percent=False):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    positions = np.arange(len(FRIENDLY))
    heights = [np.nan if value is None else value for value in values]
    bars = ax.bar(positions, heights, color=["#5875A4", "#72A0C1", "#8C9F67", "#BB765C"])
    ax.set_xticks(positions, FRIENDLY, rotation=12, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    if percent:
        ax.set_ylim(0, 1.04)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    for bar, value in zip(bars, values):
        if value is not None:
            label = f"{value:.1%}" if percent else f"{value:.2f}"
            ax.annotate(label, (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=280, bbox_inches="tight")
    plt.close(fig)


def draw_cm(result, path):
    cm = np.asarray(result["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    labels = [name.replace("___", "\n") for name in CLASS_NAMES]
    ax.set_xticks(range(10), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(10), labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Hybrid — PlantVillage test confusion matrix (verified run)")
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=280, bbox_inches="tight")
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audit_dir = ROOT / "results/audit"
    figures_dir = ROOT / "results/final_figures"
    domain_dir = ROOT / "results/domain_shift"
    audit_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pv_results = {}
    resources = {}
    checks = {}
    for key in MODEL_SPECS:
        print(f"Verifying {key} on PlantVillage test split using {device}…", flush=True)
        result = verify_plantvillage(key, device)
        pv_results[key] = result
        saved = read_metrics(PV_SOURCE_FILES[key])
        checks[key] = {
            metric: {
                "saved": saved[metric], "verified": result["metrics"][metric],
                "matches_within_1e-6": abs(float(saved[metric]) - result["metrics"][metric]) <= 1e-6,
            }
            for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
        }
        print(json.dumps(result["metrics"], indent=2), flush=True)
        (audit_dir / f"{key}_plantvillage_verified.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(f"Measuring model resources for {key}…", flush=True)
        resources[key] = measure_resources(key, device)

    external_results = {}
    for key in MODEL_SPECS:
        path = domain_dir / f"{key}_plantdoc_test_results.json"
        if path.exists():
            external_results[key] = json.loads(path.read_text(encoding="utf-8"))

    preparation_path = ROOT / "data/external/plantdoc_prepared/preparation_manifest.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8")) if preparation_path.is_file() else {}
    explanation_path = ROOT / "results/explainability/selection_summary.json"
    explanation_summary = json.loads(explanation_path.read_text(encoding="utf-8")) if explanation_path.is_file() else None
    integrity_status = verify_checkpoint()

    comparison = []
    for key in MODEL_SPECS:
        pv = pv_results[key]["metrics"]
        ext_result = external_results.get(key)
        ext = ext_result.get("metrics") if ext_result else None
        mapped_names = ext_result["mapped_classes"] if ext_result else []
        pv_overlap = metrics_for_mapped_subset(
            pv_results[key]["confusion_matrix"], CLASS_NAMES, mapped_names
        ) if ext_result else None
        comparison.append({
            "model_key": key,
            "model": MODEL_SPECS[key]["name"],
            "plantvillage": pv,
            "plantvillage_mapped_overlap": pv_overlap,
            "external": ext,
            "domain_gap": {
                metric: (pv_overlap[pv_key] - ext[ext_key]) if ext else None
                for metric, pv_key, ext_key in (
                    ("accuracy", "accuracy", "accuracy"),
                    ("macro_f1", "f1_macro", "f1_macro"),
                )
            },
            **resources[key],
        })

    final = {
        "project": "LeafGuard",
        "research_question": "Does heterogeneous MobileNetV3 + DeiT-Tiny feature fusion improve robustness/generalization from controlled PlantVillage images to more realistic field imagery, and can Grad-CAM++ provide useful visual explanations?",
        "dataset": {
            "plantvillage_split_counts_reported": {"train": 7647, "validation": 845, "test": 2054},
            "plantvillage_classes": CLASS_NAMES,
            "external_dataset": "PlantDoc" if external_results else None,
            "external_images": next(iter(external_results.values()))["sample_count"] if external_results else 0,
            "external_classes": next(iter(external_results.values()))["mapped_classes"] if external_results else [],
            "external_evaluation_completed": bool(external_results),
            "external_source": preparation.get("source_url"),
            "external_license": preparation.get("license"),
            "external_source_commit": preparation.get("source_commit"),
            "external_source_test_images": preparation.get("source_test_image_count"),
            "external_mapped_bytes": preparation.get("source_test_bytes_mapped"),
            "external_mapped_class_counts": preparation.get("mapped_class_counts", {}),
        },
        "plantvillage_verified_results": {key: value["metrics"] for key, value in pv_results.items()},
        "plantvillage_full_results": pv_results,
        "saved_vs_verified_consistency": checks,
        "resource_measurements": resources,
        "runtime": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "resource_latency_warmup_iterations": 8,
            "resource_latency_measurement_iterations": 60,
            "resource_latency_measurement_repeats": 3,
        },
        "model_integrity": integrity_status,
        "explainability": {
            "method": "Grad-CAM++ for MobileNet final convolution and DeiT final-block norm1 patch tokens; branch maps are separately normalized and averaged.",
            "examples": explanation_summary,
            "quantitative_localization": "Not measured: the evaluated cropped classification images have no verified paired region annotations. The separate PlantDoc object-detection release has leaf boxes, but no image correspondence to this pinned classification test subset was established; those boxes are not disease-lesion masks.",
        },
        "model_comparison": comparison,
        "external_results": {key: value for key, value in external_results.items()},
        "additional_hybrid_clahe": {
            "status": "not_trained",
            "reason": "The brief made this optional and required a stable, computationally reasonable implementation; no extra training was authorized or needed for the core four-checkpoint comparison.",
        },
        "limitations": [
            "PlantVillage evaluation measures performance on a controlled image domain.",
            "The PlantDoc comparison uses only 83 images across nine mapped classes; it does not establish general field robustness.",
            "Latency is the median of three batch-one model-forward trials after warmup; image decoding and preprocessing are excluded, and results vary with runtime and hardware.",
            "Profiler FLOPs are approximate and omit unsupported operators.",
            "Grad-CAM++ is a class-discriminative visualization, not a causal explanation.",
            "PlantDoc images are internet-sourced and class mappings are restricted to explicit compatible labels.",
        ],
    }
    (ROOT / "results/final_results.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    (audit_dir / "plantvillage_saved_vs_verified.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    keys = list(MODEL_SPECS)
    draw_bar([pv_results[k]["metrics"]["accuracy"] for k in keys],
             "PlantVillage test accuracy", "Accuracy", figures_dir / "model_accuracy_comparison.png", True)
    draw_bar([pv_results[k]["metrics"]["f1_macro"] for k in keys],
             "PlantVillage test macro F1", "Macro F1", figures_dir / "macro_f1_comparison.png", True)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    positions = np.arange(4)
    if external_results:
        paired_pv = [row["plantvillage_mapped_overlap"]["accuracy"] for row in comparison]
        ext_acc = [external_results[k]["metrics"]["accuracy"] for k in keys]
        width = 0.36
        ax.bar(positions - width/2, paired_pv, width, label="PlantVillage (same 9 classes)", color="#5875A4")
        ax.bar(positions + width/2, ext_acc, width, label="PlantDoc mapped test", color="#BB765C")
        ax.set_xticks(positions, FRIENDLY, rotation=12, ha="right")
        ax.set_ylim(0, 1.18)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.set_ylabel("Accuracy")
        ax.set_title("PlantVillage vs PlantDoc accuracy on the same mapped classes")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / "plantvillage_vs_external_accuracy.png", dpi=280, bbox_inches="tight")
        plt.close(fig)
        paired_pv_f1 = [row["plantvillage_mapped_overlap"]["f1_macro"] for row in comparison]
        external_f1 = [external_results[k]["metrics"]["f1_macro"] for k in keys]
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.bar(positions - width/2, paired_pv_f1, width, label="PlantVillage (same 9 classes)", color="#5875A4")
        ax.bar(positions + width/2, external_f1, width, label="PlantDoc mapped test", color="#BB765C")
        ax.set_xticks(positions, FRIENDLY, rotation=12, ha="right")
        ax.set_ylim(0, 1.04)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.set_ylabel("Macro F1")
        ax.set_title("PlantVillage vs PlantDoc macro F1 on the same mapped classes")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / "plantvillage_vs_external_macro_f1.png", dpi=280, bbox_inches="tight")
        plt.close(fig)
        draw_bar(paired_pv_f1, "PlantVillage macro F1 on PlantDoc-overlapping classes", "Macro F1",
                 figures_dir / "plantvillage_mapped_macro_f1.png", True)
        gaps = [row["domain_gap"]["accuracy"] for row in comparison]
        draw_bar(gaps, "Accuracy domain gap (PlantVillage − PlantDoc)", "Absolute accuracy gap", figures_dir / "domain_gap_accuracy.png")
        f1_gaps = [row["domain_gap"]["macro_f1"] for row in comparison]
        draw_bar(f1_gaps, "Macro F1 domain gap (PlantVillage − PlantDoc)", "Absolute macro F1 gap", figures_dir / "domain_gap_macro_f1.png")
    else:
        ax.text(0.5, 0.5, "PlantDoc not yet acquired/prepared\nExternal comparison is not available",
                ha="center", va="center", transform=ax.transAxes, fontsize=13)
        ax.set_axis_off()
        ax.set_title("PlantVillage vs external-domain accuracy")
        fig.tight_layout()
        fig.savefig(figures_dir / "plantvillage_vs_external_accuracy_pending.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
    params = [resources[k]["parameter_count"] / 1e6 for k in keys]
    draw_bar(params, "Trainable and frozen parameters by model", "Parameters (millions)", figures_dir / "parameter_count_comparison.png")
    sizes = [resources[k]["checkpoint_size_mib"] for k in keys]
    draw_bar(sizes, "Checkpoint file size", "MiB", figures_dir / "model_size_comparison.png")
    draw_cm(pv_results["hybrid"], figures_dir / "hybrid_plantvillage_confusion_matrix.png")

    header = "| Model | PV accuracy (10) | PV macro P (10) | PV macro R (10) | PV macro F1 (10) | PV accuracy (mapped 9) | PV macro F1 (mapped 9) | PlantDoc accuracy (9) | PlantDoc macro P (9) | PlantDoc macro R (9) | PlantDoc macro F1 (9) | Accuracy gap* | Macro F1 gap* | Parameters | Checkpoint MiB | Forward latency ms/img | Approx FLOPs/img |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header]
    for row in comparison:
        pv = row["plantvillage"]
        overlap = row["plantvillage_mapped_overlap"] or {}
        ext = row["external"] or {}
        fmt = lambda value, digits=4: "—" if value is None else f"{value:.{digits}f}"
        lines.append(
            f"| {row['model']} | {fmt(pv['accuracy'])} | {fmt(pv['precision_macro'])} | {fmt(pv['recall_macro'])} | {fmt(pv['f1_macro'])} | {fmt(overlap.get('accuracy'))} | {fmt(overlap.get('f1_macro'))} | {fmt(ext.get('accuracy'))} | {fmt(ext.get('precision_macro'))} | {fmt(ext.get('recall_macro'))} | {fmt(ext.get('f1_macro'))} | {fmt(row['domain_gap']['accuracy'])} | {fmt(row['domain_gap']['macro_f1'])} | {row['parameter_count']:,} | {row['checkpoint_size_mib']:.2f} | {row['inference_latency_ms_batch1']:.2f} | {row['approximate_flops_batch1'] if row['approximate_flops_batch1'] is not None else '—'} |"
        )
    table_note = (
        f"PlantDoc mapped test subset: {final['dataset']['external_images']} images across "
        f"{len(final['dataset']['external_classes'])} classes. *Domain gaps use PlantVillage metrics recomputed on the same nine classes; predictions into the model's other output classes count as errors. FLOPs are profiler estimates and may omit operators. "
        f"Latency is the median of three batch-one model-forward trials after eight warmups on {device}; image decoding and preprocessing are excluded."
        if external_results else
        f"External results and gaps are blank until PlantDoc is prepared and evaluated. FLOPs are profiler estimates and may omit operators. Latency is the median of three batch-one model-forward trials after eight warmups on {device}, excluding image decoding and preprocessing."
    )
    table = "\n".join(lines) + "\n\n" + table_note + "\n"
    (figures_dir / "experiment_table.md").write_text(table, encoding="utf-8")

    md = [
        "# LeafGuard final results",
        "",
        "## Dataset",
        "",
        "PlantVillage: 10 selected classes; reported split counts 7,647 train, 845 validation, and 2,054 test. The held-out test set was re-evaluated from the existing checkpoints. " + (
            f"PlantDoc: {final['dataset']['external_source_test_images']} upstream test images were listed, of which {final['dataset']['external_images']} mapped images from nine compatible classes were prepared from pinned commit {final['dataset']['external_source_commit'][:12]}. The mapped subset totals {final['dataset']['external_mapped_bytes']:,} bytes."
            if external_results else "PlantDoc files were unavailable; external metrics are pending."
        ),
        "",
        "## Experimental setup",
        "",
        "All four frozen checkpoints were loaded with their original 224×224 preprocessing; the CLAHE checkpoint uses LAB lightness CLAHE (clip limit 2.0, 8×8 tiles). No training was run. Verified outputs are saved in `results/audit/`.",
        "",
        "## Baseline, CLAHE, DeiT, and hybrid results",
        "",
        table,
        "## External-domain results and domain shift",
        "",
        (
            "Accuracy and macro-F1 domain gaps compare the same nine mapped classes (PlantVillage "
            f"n={comparison[3]['plantvillage_mapped_overlap']['total']} vs PlantDoc n={external_results['hybrid']['sample_count']}) and are calculated as PlantVillage minus PlantDoc. "
            f"For the hybrid, the gaps are {comparison[3]['domain_gap']['accuracy']:.4f} accuracy and "
            f"{comparison[3]['domain_gap']['macro_f1']:.4f} macro F1. DeiT-Tiny has the best external accuracy "
            f"({external_results['deit_tiny']['metrics']['accuracy']:.4f}) and macro F1 "
            f"({external_results['deit_tiny']['metrics']['f1_macro']:.4f}) among these four checkpoints. "
            "These results do not support a claim that the hybrid is more robust than DeiT-Tiny; class supports are small."
            if external_results else
            "PlantDoc could not be evaluated because no local source dataset was available. No accuracy or macro F1 domain gaps are reported."
        ),
        "",
        "## Explainability analysis",
        "",
        "Grad-CAM++ is implemented for the MobileNet convolutional branch and DeiT patch-token branch. Branch maps are normalized independently and averaged for the combined display. The output is qualitative attribution; PlantDoc's classification split does not provide pixel masks for a localization metric.",
        "",
        "## Ablation and limitations",
        "",
        "The four checkpoints form the available model comparison. An additional CLAHE hybrid was not trained. Parameter count, file size, batch-one latency and approximate profiler FLOPs appear above. The external mapped subset is limited to 83 images and sparse classes; results should not be generalized beyond this evaluation.",
        "",
        "## Conclusions supported by current data",
        "",
        (
            "The saved PlantVillage scores were reproduced. On the mapped PlantDoc subset, DeiT-Tiny outperforms the hybrid in both accuracy and macro F1; therefore current evidence does not show superior hybrid robustness."
            if external_results else
            "The checkpoints achieve the re-measured PlantVillage scores recorded in `results/audit/`. The saved results should be treated as confirmed only where the saved-versus-verified checks match."
        ),
        "",
    ]
    (ROOT / "results/final_results.md").write_text("\n".join(md), encoding="utf-8")
    print("Final results and figures saved.")


if __name__ == "__main__":
    main()
