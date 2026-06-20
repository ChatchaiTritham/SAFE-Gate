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

# Curated set MUST match what evaluation/generate_figures_from_results.py actually
# produces (the pie-chart "safety_performance" was retired; the headline is now the
# "safety_and_ablation" bar+CI panel). No supplementary stems are emitted by the
# active generator, so none are listed here.
SOURCE_SCRIPT = "evaluation/generate_figures_from_results.py"
FIGURES = [
    {
        "figure_id": "SAFE-F1",
        "stem": "safety_and_ablation",
        "role": "manuscript",
        "source_data": "results/summary.json",
        "caption": "Safety performance and gate ablation. Critical-case sensitivity is shown as a bar with a Wilson 95% binomial confidence interval [97.9%, 100.0%] for the zero-event estimate (175/175 caught, 0 false discharges); R5 discharge specificity, over-triage rate, and macro-F1 are reported alongside. Right: over-triage and macro-F1 under single-gate ablation.",
        "article_section": "Safety performance",
    },
    {
        "figure_id": "SAFE-F2",
        "stem": "baseline_comparison",
        "role": "manuscript",
        "source_data": "results/baseline_comparison.csv",
        "caption": "Cleveland dot plot of the safety-vs-accuracy trade-off across methods (critical sensitivity vs overall accuracy, test fold, seed 42). SAFE-Gate attains maximal critical sensitivity at the lowest overall accuracy.",
        "article_section": "Baseline comparison",
    },
    {
        "figure_id": "SAFE-F3",
        "stem": "confusion_matrix",
        "role": "manuscript",
        "source_data": "results/confusion_matrix.csv",
        "caption": "Risk-tier confusion matrix (R1-R5, test fold, seed 42). No critical case is routed to discharge. The reproduced ACWCM concentrates predictions on R1 and R4; R2, R3 and R5 receive zero recall (R3 is never predicted), so the system is not a functioning five-tier classifier - annotated on the figure.",
        "article_section": "Classification performance",
    },
    {
        "figure_id": "SAFE-F4",
        "stem": "per_class_metrics",
        "role": "manuscript",
        "source_data": "results/tier_metrics.csv",
        "caption": "Per-tier precision, recall, and F1 with Wilson 95% confidence intervals on recall (test fold, seed 42). R2, R3 and R5 collapse to recall = 0 (flagged on the figure); only R1 and R4 are recovered.",
        "article_section": "Classification performance",
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
                    "source_script": SOURCE_SCRIPT,
                    "source_data": item["source_data"],
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
