"""Create deterministic contact sheets for manual external-image review."""
import argparse
import json
import math
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageFont

from src.inference.common import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/external/plantdoc_prepared/test")
    parser.add_argument("--manifest", type=Path, help="Mapped ZIP image inventory JSON")
    parser.add_argument("--archive", type=Path, help="Source ZIP used with --manifest")
    parser.add_argument("--dataset-slug", default="plantdoc")
    parser.add_argument("--output", type=Path, default=ROOT / "results/plots/domain_shift/visual_review")
    parser.add_argument("--per-page", type=int, default=16)
    args = parser.parse_args()
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    if bool(args.manifest) != bool(args.archive):
        raise ValueError("--manifest and --archive must be supplied together")
    records = []
    if args.manifest:
        payload = json.loads(args.manifest.read_text(encoding="utf-8"))
        records = sorted(payload["eligible_records"], key=lambda row: row["image_id"])
        archive = ZipFile(args.archive)
        source_items = [(row["image_id"], row["image_member"], row.get("width_px"), row.get("height_px"), row["true_label"])
                        for row in records]
    else:
        files = sorted(path for path in args.dataset.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        source_items = [(path.relative_to(args.dataset).as_posix(), path, None, None, None) for path in files]
    if not source_items:
        raise RuntimeError(f"No image files found in {args.dataset}")
    args.output.mkdir(parents=True, exist_ok=True)
    cols = 4
    tile_w, tile_h = 300, 270
    thumb_w, thumb_h = 286, 220
    font = ImageFont.load_default()
    manifest = []
    page_count = math.ceil(len(source_items) / args.per_page)
    for page_index in range(page_count):
        page_files = source_items[page_index * args.per_page:(page_index + 1) * args.per_page]
        rows = math.ceil(len(page_files) / cols)
        sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, (image_id, source, width, height, label) in enumerate(page_files):
            if args.manifest:
                image_context = Image.open(BytesIO(archive.read(source)))
            else:
                image_context = Image.open(source)
            with image_context as image:
                image = image.convert("RGB")
                width, height = image.size
                image.thumbnail((thumb_w, thumb_h))
                record = {
                    "image_id": image_id,
                    "width": width,
                    "height": height,
                    "minimum_dimension_px": min(width, height),
                    "contact_sheet_page": page_index + 1,
                    "contact_sheet_slot": offset + 1,
                }
                if label is not None:
                    record["true_label"] = label
                manifest.append(record)
            col, row = offset % cols, offset // cols
            x0, y0 = col * tile_w, row * tile_h
            x = x0 + (tile_w - image.width) // 2
            y = y0 + 3
            sheet.paste(image, (x, y))
            draw.rectangle((x0 + 4, y0 + 3, x0 + tile_w - 4, y0 + thumb_h + 8), outline="#888888")
            caption = f"{record['image_id']}\n{width} x {height}px"
            draw.multiline_text((x0 + 7, y0 + thumb_h + 13), caption, fill="black", font=font, spacing=2)
        out_path = args.output / f"{args.dataset_slug}_visual_review_{page_index + 1:02d}.png"
        sheet.save(out_path, optimize=True)
    if args.manifest:
        archive.close()
    (args.output / "contact_sheet_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {page_count} contact sheets for {len(source_items)} images under {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
