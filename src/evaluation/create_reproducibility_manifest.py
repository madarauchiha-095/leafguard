"""Write a reproducibility manifest from the current runtime and frozen artifacts."""
import hashlib
import importlib.metadata as metadata
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision
import timm

from src.inference.common import CLASS_NAMES, MODEL_SPECS, ROOT, IMAGENET_MEAN, IMAGENET_STD


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def requirements_versions():
    requirements = ROOT / "requirements.txt"
    result = {}
    raw = requirements.read_bytes()
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        decoded = raw.decode("utf-16")
    for line in decoded.splitlines():
        text = line.strip()
        if not text or text.startswith("#") or text.startswith(("-", "git+", "http")):
            continue
        # This repository's requirements file uses exact pins. Avoid importing
        # packaging just to report whether the active environment matches them.
        if "==" in text:
            name, pinned = (part.strip() for part in text.split("==", 1))
        else:
            name, pinned = text, None
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        if installed is None:
            status = "NOT_INSTALLED"
            matches = False
        elif pinned is not None:
            matches = installed == pinned
            status = "MATCH" if matches else "MISMATCH"
        else:
            matches = None
            status = "UNPINNED"
        result[name] = {
            "requirements_entry": text,
            "installed_version": installed,
            "requirement_status": status,
            "requirement_matches_installed": matches,
        }
    return result


def main():
    integrity = json.loads((ROOT / "configs/model_integrity.json").read_text(encoding="utf-8"))
    source = json.loads((ROOT / "data/external/plantdoc_source_subset/source_manifest.json").read_text(encoding="utf-8"))
    prepared = json.loads((ROOT / "data/external/plantdoc_prepared/preparation_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "results/audit/reproducibility_audit.json").read_text(encoding="utf-8"))
    plantseg_audit = json.loads((ROOT / "results/audit/plantseg_v7_archive_audit.json").read_text(encoding="utf-8"))
    plantseg_evaluation_audit = json.loads((ROOT / "results/audit/plantseg_v7_evaluation_audit.json").read_text(encoding="utf-8"))
    final = json.loads((ROOT / "results/final_results.json").read_text(encoding="utf-8"))

    checkpoints = {}
    for key, spec in MODEL_SPECS.items():
        path = spec["checkpoint"]
        checkpoints[key] = {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "parameters": final["resource_measurements"][key]["parameter_count"],
            "preprocessing": {
                "resize": [224, 224],
                "normalization_mean": list(IMAGENET_MEAN),
                "normalization_std": list(IMAGENET_STD),
                "clahe": bool(spec["clahe"]),
                "clahe_clip_limit": 2.0 if spec["clahe"] else None,
                "clahe_tile_grid_size": [8, 8] if spec["clahe"] else None,
            },
        }

    dependency_versions = requirements_versions()
    dependency_mismatches = {
        name: item for name, item in dependency_versions.items()
        if item["requirement_status"] in {"MISMATCH", "NOT_INSTALLED", "NOT_PARSED"}
    }
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project": "LeafGuard",
        "project_revision": "unavailable; workspace has no .git metadata",
        "environment": {
            "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "pytorch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
            "cuda_available": torch.cuda.is_available(),
            "pytorch_cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "dependencies_from_requirements_txt": dependency_versions,
            "dependency_mismatches": dependency_mismatches,
        },
        "datasets": {
            "plantvillage_derived": {
                "source": "Existing processed project subset; original acquisition URL/version is not recorded in this workspace.",
                "license": "Not independently established from available project metadata.",
                "classes": CLASS_NAMES,
                "split_counts": {"train": 7647, "validation": 845, "test": 2054},
                "semantic_label_review": "Folder labels and class order mechanically checked; no independent relabeling.",
            },
            "plantdoc": {
                "source_url": source["source_url"],
                "source_commit": source["source_commit"],
                "license_statement": source["source_license"],
                "evaluated_split": source["source_split"],
                "upstream_test_images": source["source_test_image_count"],
                "mapped_images": prepared["source_test_images_mapped"],
                "mapped_class_counts": prepared["mapped_class_counts"],
                "class_mapping": prepared["class_mapping"],
                "prepared_source_blob_sha256_verified": audit["findings"]["plantdoc_inventory"]["provenance_hashes"]["status"] == "VERIFIED",
            },
            "plantseg_v7": {
                "source_url": plantseg_audit["source"],
                "release": plantseg_audit["release"],
                "archive_path": plantseg_audit["archive"]["path"],
                "archive_bytes": plantseg_audit["archive"]["bytes"],
                "archive_md5": plantseg_audit["archive"]["actual_md5"],
                "archive_sha256": plantseg_audit["archive"]["sha256"],
                "official_record_md5_match": plantseg_audit["archive"]["md5_match"],
                "source_split_counts": plantseg_audit["metadata"]["split_counts"],
                "evaluated_split": "official Test split only; no training or validation images",
                "license_selection": "CC-BY-NC rows only; two CC0 mapped candidates excluded before evaluation to keep a single license selection rule; no image redistribution",
                "mapped_images": plantseg_audit["mapping"]["eligible_test_records_before_duplicate_removal"],
                "mapped_class_counts": plantseg_audit["mapping"]["eligible_test_class_counts_before_duplicate_removal"],
                "class_mapping": plantseg_audit["mapping"]["included_source_disease_to_leafguard_label"],
                "overlap_screen": plantseg_audit["overlap_screen"],
                "archive_audit_status": plantseg_audit["audit_status"],
                "prediction_evaluation_audit_status": plantseg_evaluation_audit["audit_status"],
            },
        },
        "checkpoints": checkpoints,
        "random_seeds": {
            "mobilenet_training": None,
            "mobilenet_clahe_training": None,
            "deit_tiny_training": 42,
            "hybrid_training": 42,
            "bootstrap": 20260924,
            "note": "The two MobileNet training scripts set no explicit seed. Existing checkpoints were not retrained.",
        },
        "evaluation_settings": {
            "input_size": [224, 224],
            "plantvillage_batch_size": 32,
            "plantdoc_batch_size": 16,
            "plantseg_v7_batch_size": 16,
            "resource_timing_batch_size": 1,
            "resource_timing_warmups": 8,
            "resource_timing_repeats": 3,
            "resource_timing_iterations_per_repeat": 60,
            "resource_timing_method": "CUDA events on GPU; median model-forward time only, excluding image decode and preprocessing.",
            "external_confidence_high_score_threshold": 0.90,
            "paired_comparison_bootstrap_replicates": 10000,
            "paired_comparison_bootstrap_seed": 20260924,
            "per_model_and_domain_gap_bootstrap_replicates": 10000,
            "per_model_and_domain_gap_bootstrap_seed": 20260924,
            "classes": CLASS_NAMES,
            "unsupported_output_predictions": "Counted as errors; full ten-output confusion matrices retained.",
        },
        "commands": [
            r".\.venv\Scripts\python.exe -m src.evaluation.reproducibility_audit",
            r".\.venv\Scripts\python.exe -m src.evaluation.domain_shift",
            r".\.venv\Scripts\python.exe -m src.evaluation.domain_shift --dataset data/processed/test --output results/domain_shift/plantvillage_per_image --dataset-name 'PlantVillage held-out test split' --dataset-slug plantvillage_test --batch-size 32",
            r".\.venv\Scripts\python.exe -m src.evaluation.domain_shift --manifest results/audit/plantseg_v7_archive_audit.json --archive data/external/plantseg_v7_source.zip --output results/domain_shift/plantseg_v7 --dataset-name 'PlantSeg v7 mapped test split (CC-BY-NC only)' --dataset-slug plantseg_v7",
            r".\.venv\Scripts\python.exe -m src.evaluation.summarize_external_results --results-dir results/domain_shift --dataset-slug plantdoc --dataset-name 'PlantDoc mapped test split'",
            r".\.venv\Scripts\python.exe -m src.evaluation.summarize_external_results --results-dir results/domain_shift/plantseg_v7 --dataset-slug plantseg_v7 --dataset-name 'PlantSeg v7 mapped test split (CC-BY-NC only)'",
            r".\.venv\Scripts\python.exe -m src.evaluation.analyze_confidence --predictions results/domain_shift/plantdoc_predictions.csv --external-results results/domain_shift --external-slug plantdoc --output-json results/domain_shift/confidence_analysis.json --plot-dir results/plots/domain_shift",
            r".\.venv\Scripts\python.exe -m src.evaluation.analyze_confidence --predictions results/domain_shift/plantseg_v7/plantseg_v7_predictions.csv --external-results results/domain_shift/plantseg_v7 --external-slug plantseg_v7 --output-json results/domain_shift/plantseg_v7/confidence_analysis.json --plot-dir results/plots/domain_shift/plantseg_v7",
            r".\.venv\Scripts\python.exe -m src.evaluation.verify_external_results --results-dir results/domain_shift/plantseg_v7 --audit-manifest results/audit/plantseg_v7_archive_audit.json --checkpoint-manifest research/reproducibility_manifest.json --dataset-slug plantseg_v7 --output results/audit/plantseg_v7_evaluation_audit.json",
            r".\.venv\Scripts\python.exe -m src.evaluation.finalize_results",
            r".\.venv\Scripts\python.exe -m src.evaluation.uncertainty",
            r".\.venv\Scripts\python.exe -m src.explainability.generate_examples",
            r".\.venv\Scripts\python.exe -m src.explainability.generate_paired_examples",
            r".\.venv\Scripts\python.exe -m src.evaluation.build_final_research_artifacts",
        ],
        "integrity": {
            "hybrid_expected_sha256": integrity["sha256"],
            "verification_status": audit["findings"]["checkpoint_and_resource_integrity"]["hybrid_expected_sha256"]["status"],
            "scope": "Local digest change detection, not signed provenance.",
        },
        "reproducibility_warnings": [
            "PlantVillage source/version/license metadata was not recorded in the workspace.",
            "MobileNetV3 and MobileNetV3+CLAHE training scripts do not set explicit random seeds.",
            "PlantDoc has only 83 mapped test examples; label mapping has not been independently expert-adjudicated.",
            "PlantSeg v7 has 143 mapped test images in seven classes; one class has n=9, labels have not been independently expert-adjudicated, and visual review found mixed leaf/fruit/whole-plant content plus watermarks/graphics.",
            "GPU latency shows cross-run drift even though individual trial sets are mostly stable.",
        ],
    }
    out = ROOT / "research/reproducibility_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)}")

    report = [
        "# Reproducibility Environment Report", "",
        f"Manifest generated at {manifest['generated_utc']}.", "",
        "## Runtime", "",
        f"- Python: `{manifest['environment']['python'].split()[0]}` ({manifest['environment']['python_implementation']})",
        f"- OS/architecture: {manifest['environment']['os']} {manifest['environment']['os_release']} / {manifest['environment']['architecture']}",
        f"- PyTorch / torchvision / timm: {torch.__version__} / {torchvision.__version__} / {timm.__version__}",
        f"- CUDA: `{torch.version.cuda}`; cuDNN: `{torch.backends.cudnn.version()}`; GPU: `{manifest['environment']['gpu']}`",
        f"- Dependency pins checked: {len(dependency_versions) - len(dependency_mismatches)}/{len(dependency_versions)} match the installed environment.",
        f"- Mismatches and missing packages: `{json.dumps(dependency_mismatches, ensure_ascii=False) if dependency_mismatches else 'none'}`.",
        f"- The complete declared and installed package versions are recorded in `{out.relative_to(ROOT).as_posix()}`.",
        "- Workspace Git revision is unavailable because the project directory has no `.git` metadata.", "",
        "## Data and model provenance", "",
        "- PlantVillage-derived source provenance/license were not available in project metadata; this is an unresolved reproducibility warning.",
        f"- PlantDoc: commit `{source['source_commit']}`, test split only, {prepared['source_test_images_mapped']} mapped images in {len(prepared['mapped_class_counts'])} classes; prepared file hashes match the pinned source manifest.",
        f"- PlantSeg v7: official Zenodo record, {plantseg_audit['mapping']['eligible_test_records_before_duplicate_removal']} CC-BY-NC-only mapped test images in {len(plantseg_audit['mapping']['eligible_test_class_counts_before_duplicate_removal'])} classes; archive and prediction audit statuses are {plantseg_audit['audit_status']} / {plantseg_evaluation_audit['audit_status']}.",
        "- All four checkpoint digests, byte sizes, parameter counts, and preprocessing settings are listed in the JSON manifest.",
        "- The two DeiT-containing training scripts set PyTorch/CUDA seeds to 42. The MobileNet scripts did not set explicit seeds; no checkpoint was retrained.", "",
        "## Evaluation reproducibility", "",
        "- PlantVillage verification uses batch 32; PlantDoc and PlantSeg use batch 16; resource measurement is batch 1.",
        "- Inputs resize to 224 by 224 and use ImageNet normalization. CLAHE is applied only for the CLAHE checkpoint (LAB L, clip 2.0, 8x8 grid).",
        "- Resource latency is median CUDA-event model-forward time from three 60-forward trials after eight warmups; preprocessing and decode are excluded.",
        "- Reproduction commands are included in the manifest. Training commands are intentionally omitted: the verified study reuses frozen checkpoints.", "",
        "## Limits", "",
        "The requirements file does not exactly match the active environment for four pinned packages (`datasets`, `dill`, `fsspec`, and `multiprocess`); the exact required and installed versions are listed in the manifest. The evaluated scripts completed in this environment, but rebuilding from requirements alone would not recreate the audited runtime exactly.",
        "The exact PlantVillage source cannot be reconstructed from the current metadata, two training runs lack recorded seeds, external labels require expert review, PlantSeg's mapped subset has mixed plant-organ/graphic content and only seven disease classes, and GPU timing varies across separate runs. These limits do not invalidate the recorded held-out inference calculations, but they constrain exact training reproduction and broad generalization claims.", "",
    ]
    report_path = ROOT / "research/reproducibility_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
