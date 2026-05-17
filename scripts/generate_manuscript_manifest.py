"""Create SAFE-Gate curated manuscript figure manifest and visual QA sheet.

SAFE-Gate already ships manuscript-ready figure exports under
``evaluation/manuscript_figures``. This script records the curated set in a
top-level manifest and creates a contact sheet for visual QA without rewriting
the figure-generation pipeline.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE_DIR = ROOT / "evaluation" / "manuscript_figures"
DEFAULT_MANIFEST = ROOT / "FIGURE_MANIFEST.csv"
DPI = 300

FIGURES = [
    {
        "figure_id": "SAFE-F1",
        "stem": "safety_performance",
        "role": "manuscript",
        "caption": "Safety performance summary for critical-case detection, abstention behavior, risk distribution, and sensitivity by tier.",
        "article_section": "Safety performance",
    },
    {
        "figure_id": "SAFE-F2",
        "stem": "baseline_comparison",
        "role": "manuscript",
        "caption": "Baseline comparison showing SAFE-Gate against conventional triage and ensemble baselines.",
        "article_section": "Baseline comparison",
    },
    {
        "figure_id": "SAFE-F3",
        "stem": "confusion_matrix",
        "role": "manuscript",
        "caption": "Risk-tier confusion matrix used to assess classification behavior across R1-R5 tiers.",
        "article_section": "Classification performance",
    },
    {
        "figure_id": "SAFE-F4",
        "stem": "per_class_metrics",
        "role": "manuscript",
        "caption": "Per-class precision, recall, and F1-score by risk tier.",
        "article_section": "Classification performance",
    },
    {
        "figure_id": "SAFE-F5",
        "stem": "risk_distribution",
        "role": "supplementary",
        "caption": "Observed risk-tier distribution used to contextualize evaluation results.",
        "article_section": "Supplementary cohort profile",
    },
    {
        "figure_id": "SAFE-F6",
        "stem": "support_distribution",
        "role": "supplementary",
        "caption": "Support distribution across risk tiers for manuscript evaluation figures.",
        "article_section": "Supplementary cohort profile",
    },
]


def write_manifest(figure_dir: Path, manifest_path: Path) -> None:
    fieldnames = [
        "figure_id",
        "role",
        "png",
        "pdf",
        "source_script",
        "source_data",
        "caption",
        "article_section",
        "generated_at",
        "dpi",
    ]
    generated_at = datetime.now().isoformat(timespec="seconds")
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in FIGURES:
            png_path = figure_dir / f"{item['stem']}.png"
            pdf_path = figure_dir / f"{item['stem']}.pdf"
            if not png_path.exists() or not pdf_path.exists():
                raise FileNotFoundError(f"Missing figure pair for {item['stem']}")
            writer.writerow(
                {
                    "figure_id": item["figure_id"],
                    "role": item["role"],
                    "png": str(png_path.relative_to(ROOT)),
                    "pdf": str(pdf_path.relative_to(ROOT)),
                    "source_script": "evaluation/create_manuscript_figures.py",
                    "source_data": "evaluation/create_manuscript_figures.py",
                    "caption": item["caption"],
                    "article_section": item["article_section"],
                    "generated_at": generated_at,
                    "dpi": str(DPI),
                }
            )


def make_contact_sheet(figure_dir: Path) -> Path:
    pngs = [figure_dir / f"{item['stem']}.png" for item in FIGURES]
    thumbs = []
    for path in pngs:
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            original = thumb.size
            thumb.thumbnail((500, 340), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (540, 410), "white")
            canvas.paste(thumb, ((540 - thumb.width) // 2, 42))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), path.name, fill="black")
            draw.text((8, 384), f"{original[0]}x{original[1]}", fill="black")
            thumbs.append(canvas)

    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 540, rows * 410), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % cols) * 540, (index // cols) * 410))

    sheet_path = figure_dir / "visual_qa_contact_sheet.png"
    sheet.save(sheet_path)
    return sheet_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SAFE-Gate manuscript figure manifest")
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    write_manifest(args.figure_dir, args.manifest)
    sheet_path = make_contact_sheet(args.figure_dir)

    print(f"Wrote manifest: {args.manifest}")
    print(f"Wrote visual QA contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
