"""Audit PlantSeg v7 metadata, split, per-image terms, and source overlap."""
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image

from src.inference.common import ROOT


ARCHIVE = ROOT / "data/external/plantseg_v7_source.zip"
EXPECTED_MD5 = "9358a66dff88cdd15c4fe009763c40a3"
TARGET_MAP = {
    "apple scab": "Apple___Apple_scab",
    "apple black rot": "Apple___Black_rot",
    "grape black rot": "Grape___Black_rot",
    "potato early blight": "Potato___Early_blight",
    "potato late blight": "Potato___Late_blight",
    "tomato early blight": "Tomato___Early_blight",
    "tomato late blight": "Tomato___Late_blight",
    # Excluded: "corn rust" is not specific enough to assert common rust.
}


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phash(data):
    with Image.open(io.BytesIO(data)) as image:
        array = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    dct = cv2.dct(array)
    low = dct[:8, :8].flatten()[1:]
    median = float(np.median(low))
    value = 0
    for bit in low > median:
        value = (value << 1) | int(bit)
    return value


def hamming(a, b):
    return int((a ^ b).bit_count())


def normalized_split(value):
    value = value.strip().casefold()
    return {"training": "train", "train": "train", "validation": "val", "valid": "val",
            "val": "val", "test": "test", "testing": "test"}.get(value, value)


def external_inventory(root, dataset_name):
    files = sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    items = []
    for path in files:
        data = path.read_bytes()
        items.append({
            "image_id": path.relative_to(root).as_posix(),
            "source": dataset_name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "phash": phash(data),
        })
    return items


def main():
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"PlantSeg archive missing: {ARCHIVE}")
    archive_sha256 = file_hash(ARCHIVE)
    archive_md5 = file_hash(ARCHIVE, "md5")
    with ZipFile(ARCHIVE) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise RuntimeError(f"ZIP CRC failure at {bad_member}")
        metadata_path = "plantseg/Metadata.csv"
        with archive.open(metadata_path) as raw:
            rows = list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig")))
        member_names = set(archive.namelist())
        split_counts = Counter(normalized_split(row["Split"]) for row in rows)
        license_counts = Counter(row["License"].strip() or "[missing]" for row in rows)
        class_counts = Counter((row["Disease"].strip().casefold(), normalized_split(row["Split"])) for row in rows)
        eligible_records = []
        excluded = Counter()
        for row in rows:
            disease = row["Disease"].strip().casefold()
            split = normalized_split(row["Split"])
            if split != "test":
                continue
            if disease not in TARGET_MAP:
                excluded[f"unmapped::{disease}"] += 1
                continue
            license_text = row["License"].strip().casefold().replace("_", "-").replace(" ", "-")
            if license_text not in {"cc-by-nc", "cc-by-nc-4.0"}:
                excluded[f"license::{row['License'].strip() or '[missing]'}"] += 1
                continue
            member = f"plantseg/images/test/{row['Name']}"
            if member not in member_names:
                excluded["missing_test_image_member"] += 1
                continue
            data = archive.read(member)
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                image.verify()
            eligible_records.append({
                "image_id": f"{row['Disease'].strip().casefold().replace(' ', '_')}/{row['Name']}",
                "name": row["Name"],
                "image_member": member,
                "source_disease": row["Disease"].strip(),
                "true_label": TARGET_MAP[disease],
                "split": row["Split"].strip(),
                "license": row["License"].strip(),
                "source_url": row["URL"].strip(),
                "width_px": width,
                "height_px": height,
                "sha256": hashlib.sha256(data).hexdigest(),
                "phash": phash(data),
            })

    pv_root = ROOT / "data/processed/test"
    plantdoc_root = ROOT / "data/external/plantdoc_source_subset/test"
    pv_items = external_inventory(pv_root, "PlantVillage test")
    doc_items = external_inventory(plantdoc_root, "PlantDoc source test")
    external = pv_items + doc_items
    external_sha = {row["sha256"] for row in external}
    exact_duplicates = [
        {"plantseg_image_id": row["image_id"], "sha256": row["sha256"]}
        for row in eligible_records if row["sha256"] in external_sha
    ]
    near_duplicates = []
    if eligible_records and external:
        external_phashes = np.asarray([row["phash"] for row in external], dtype=object)
        for row in eligible_records:
            for other, other_hash in zip(external, external_phashes):
                distance = hamming(row["phash"], int(other_hash))
                if distance <= 4:
                    near_duplicates.append({
                        "plantseg_image_id": row["image_id"],
                        "other_dataset": other["source"],
                        "other_image_id": other["image_id"],
                        "phash_hamming_distance": distance,
                        "exact_sha256_match": row["sha256"] == other["sha256"],
                    })

    mapped_test_counts = Counter(record["true_label"] for record in eligible_records)
    output = {
        "audit_status": "VERIFIED" if archive_md5 == EXPECTED_MD5 and not exact_duplicates else "WARNING",
        "source": "https://zenodo.org/records/17719108",
        "release": "PlantSeg v7",
        "archive": {
            "path": ARCHIVE.relative_to(ROOT).as_posix(),
            "bytes": ARCHIVE.stat().st_size,
            "expected_md5_from_official_record": EXPECTED_MD5,
            "actual_md5": archive_md5,
            "md5_match": archive_md5 == EXPECTED_MD5,
            "sha256": archive_sha256,
            "zip_crc_check": "passed" if bad_member is None else f"failed:{bad_member}",
        },
        "metadata": {
            "file": "plantseg/Metadata.csv",
            "total_rows": len(rows),
            "split_counts": dict(split_counts),
            "license_counts": dict(license_counts),
            "image_members_present_count": sum(
                f"plantseg/images/{normalized_split(row['Split'])}/{row['Name']}" in member_names for row in rows
            ),
            "total_metadata_rows": len(rows),
        },
        "mapping": {
            "included_source_disease_to_leafguard_label": TARGET_MAP,
            "excluded_reason": "corn rust is not specific enough to map to common rust; healthy categories are absent from the latest disease-only class table.",
            "eligible_test_records_before_duplicate_removal": len(eligible_records),
            "eligible_test_class_counts_before_duplicate_removal": dict(mapped_test_counts),
            "excluded_test_records": dict(excluded),
        },
        "license_screen": {
            "allowed_metadata_values_for_noncommercial_research_screen": ["CC-BY-NC", "CC-BY-NC-4.0"],
            "filtered_records_to_these_values": True,
            "image_level_license_metadata_is_source_asserted_and_not_independently_legally_adjudicated": True,
            "no_images_were_extracted_or_redistributed": True,
        },
        "overlap_screen": {
            "external_inventory_counts": {"PlantVillage_test": len(pv_items), "PlantDoc_source_test": len(doc_items)},
            "exact_sha256_duplicates": exact_duplicates,
            "perceptual_hash_method": "32x32 grayscale resize; OpenCV DCT 8x8 low-frequency median bit hash; Hamming distance <=4 flagged for visual review.",
            "near_duplicate_flags": near_duplicates,
            "near_duplicate_flag_count": len(near_duplicates),
        },
        "eligible_records": eligible_records,
    }
    out = ROOT / "results/audit/plantseg_v7_archive_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)}")
    print(json.dumps({
        "md5_match": output["archive"]["md5_match"],
        "zip_crc_check": output["archive"]["zip_crc_check"],
        "metadata_rows": len(rows),
        "splits": dict(split_counts),
        "license_counts": dict(license_counts),
        "eligible_test_records": len(eligible_records),
        "class_counts": dict(mapped_test_counts),
        "exact_duplicates": len(exact_duplicates),
        "near_duplicate_flags": len(near_duplicates),
        "excluded": dict(excluded),
    }, indent=2))


if __name__ == "__main__":
    main()
