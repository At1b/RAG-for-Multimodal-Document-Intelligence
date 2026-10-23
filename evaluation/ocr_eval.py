"""Phase 10 OCR Evaluation and Benchmark Suite.

Evaluates Character Error Rate (CER), Word Error Rate (WER), extraction latency
(native vs 200 DPI vs 300 DPI), and table layout preservation using Tesseract.
Outputs structured JSON results and a markdown summary table.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

# Ensure repository root is on sys.path for direct script execution.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pymupdf

from rag.ingestion.image_loader import load_image
from rag.ingestion.ocr.engine import TesseractEngine
from rag.ingestion.ocr.processor import OcrProcessor
from rag.ingestion.ocr.text_layout import parse_tsv_to_text
from rag.ingestion.pdf_loader import load_pdf

FIXTURES_DIR = Path(__file__).resolve().parent / "ocr" / "fixtures"
GROUND_TRUTH_DIR = Path(__file__).resolve().parent / "ocr" / "ground_truth"
RESULTS_DIR = Path(__file__).resolve().parent / "ocr" / "results"


# --- Metrics (Levenshtein Distance) ---


def levenshtein_distance(seq1: list | str, seq2: list | str) -> int:
    """Compute the Levenshtein edit distance between two sequences."""
    if len(seq1) < len(seq2):
        return levenshtein_distance(seq2, seq1)
    if len(seq2) == 0:
        return len(seq1)

    previous_row = list(range(len(seq2) + 1))
    for i, c1 in enumerate(seq1):
        current_row = [i + 1] + [0] * len(seq2)
        for j, c2 in enumerate(seq2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row
    return previous_row[-1]


def normalize_text(text: str) -> str:
    """Normalize text per Phase 10 specification:
    NFC Unicode, replace newlines/tabs/NBSP with single space, collapse whitespace.
    Case, punctuation, and hyphens are preserved so real recognition errors count.
    """
    text = unicodedata.normalize("NFC", text)
    text = (
        text.replace("\r\n", " ")
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("\u00a0", " ")
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_cer(ref: str, hyp: str) -> float:
    """Compute Character Error Rate (CER = edit_dist / len(ref))."""
    ref_norm = normalize_text(ref)
    hyp_norm = normalize_text(hyp)
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0
    dist = levenshtein_distance(ref_norm, hyp_norm)
    return dist / len(ref_norm)


def compute_wer(ref: str, hyp: str) -> float:
    """Compute Word Error Rate (WER = edit_dist / len(ref_tokens))."""
    ref_norm = normalize_text(ref)
    hyp_norm = normalize_text(hyp)
    ref_tokens = ref_norm.split()
    hyp_tokens = hyp_norm.split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    dist = levenshtein_distance(ref_tokens, hyp_tokens)
    return dist / len(ref_tokens)


# --- Extraction Helpers ---


def extract_document_text(
    file_path: Path, processor: OcrProcessor
) -> tuple[str, list[str]]:
    """Extract full document text and per-page texts using loaders."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        doc = load_pdf(file_path, ocr=processor)
        pages = [p.content for p in doc.pages]
        return "\n\n".join(pages), pages
    elif ext in {".png", ".jpg", ".jpeg"}:
        doc = load_image(file_path, ocr=processor)
        pages = [p.content for p in doc.pages]
        return doc.pages[0].content, pages
    else:
        raise ValueError(f"Unsupported fixture extension: {ext}")


# --- Table Extraction Assessment ---


def _recognize_with_psm(png_bytes: bytes, psm: int, exe_path: str) -> str:
    """Run Tesseract with a specific Page Segmentation Mode (PSM) for evaluation."""
    cmd = [
        exe_path,
        "stdin",
        "stdout",
        "-l",
        "eng",
        "--psm",
        str(psm),
        "tsv",
    ]
    creationflags = 0
    if platform.system() == "Windows":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    res = subprocess.run(
        cmd,
        input=png_bytes,
        capture_output=True,
        timeout=30.0,
        creationflags=creationflags,
        check=False,
    )
    if res.returncode != 0:
        return ""
    text, _ = parse_tsv_to_text(res.stdout.decode("utf-8", errors="replace"))
    return text


def _tsv_with_gap_heuristic(tsv_text: str) -> str:
    """Parse TSV and insert ' | ' where word gap exceeds 2x median gap."""
    reader = csv.reader(io.StringIO(tsv_text), delimiter="\t")
    rows = list(reader)
    if len(rows) < 2:
        return ""

    header = rows[0]
    expected_header = [
        "level",
        "page_num",
        "block_num",
        "par_num",
        "line_num",
        "word_num",
        "left",
        "top",
        "width",
        "height",
        "conf",
        "text",
    ]
    if [h.strip() for h in header] != expected_header:
        return ""

    lines: dict[tuple[int, int, int], list[tuple[int, int, str]]] = {}
    for r in rows[1:]:
        if len(r) != 12:
            continue
        try:
            block = int(r[2])
            par = int(r[3])
            line = int(r[4])
            left = int(r[6])
            width = int(r[8])
            conf = float(r[10])
            word = r[11].strip()
        except (ValueError, IndexError):
            continue
        if conf < 0 or not word:
            continue
        key = (block, par, line)
        lines.setdefault(key, []).append((left, width, word))

    out_lines: list[str] = []
    for _key, words in lines.items():
        if not words:
            continue
        words.sort(key=lambda w: w[0])
        if len(words) == 1:
            out_lines.append(words[0][2])
            continue

        gaps = [
            words[i + 1][0] - (words[i][0] + words[i][1]) for i in range(len(words) - 1)
        ]
        pos_gaps = [g for g in gaps if g > 0]
        med_gap = statistics.median(pos_gaps) if pos_gaps else 10.0
        gap_threshold = max(2.0 * med_gap, 20.0)

        line_str = words[0][2]
        for i in range(len(words) - 1):
            gap = words[i + 1][0] - (words[i][0] + words[i][1])
            sep = " | " if gap >= gap_threshold else " "
            line_str += sep + words[i + 1][2]
        out_lines.append(line_str)

    return "\n".join(out_lines)


def evaluate_table_modes(
    fixture_path: Path, gt_path: Path, engine: TesseractEngine
) -> dict:
    """Evaluate table extraction across PSM 3, 4, 6 and gap heuristic.

    Measures row preservation and token-presence matching (verifying that all
    key cell tokens from the ground truth table are present in recognized text,
    rather than structural cell coordinate alignment).
    """
    doc = pymupdf.open(str(fixture_path))
    page = doc[0]
    pix = page.get_pixmap(dpi=300)
    png_bytes = pix.tobytes("png")
    doc.close()

    gt_text = gt_path.read_text(encoding="utf-8")
    # Key tokens expected across all 12 cells of the 4x3 table (with "F1 Score"
    # split into its constituent tokens).
    expected_tokens = [
        "Metric",
        "Baseline",
        "Proposed",
        "Precision",
        "0.82",
        "0.94",
        "Recall",
        "0.76",
        "0.89",
        "F1",
        "Score",
        "0.79",
        "0.91",
    ]

    exe_path = engine._resolve_executable()
    results = {}
    for psm in (3, 4, 6):
        raw_text = _recognize_with_psm(png_bytes, psm, exe_path)
        norm_raw = normalize_text(raw_text)
        token_matches = sum(1 for c in expected_tokens if c in norm_raw)
        token_presence_rate = token_matches / len(expected_tokens)
        cer = compute_cer(gt_text, raw_text)
        wer = compute_wer(gt_text, raw_text)

        # Check if rows stay on separate lines
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        same_line_rows = len(lines) == 4

        results[f"psm_{psm}"] = {
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "token_presence_rate": round(token_presence_rate, 4),
            "four_rows_preserved": same_line_rows,
            "sample_output": lines[:4],
        }

    # Evaluate gap heuristic on PSM 3 TSV output
    cmd = [exe_path, "stdin", "stdout", "-l", "eng", "--psm", "3", "tsv"]
    res = subprocess.run(
        cmd, input=png_bytes, capture_output=True, timeout=30.0, check=False
    )
    gap_text = (
        _tsv_with_gap_heuristic(res.stdout.decode("utf-8", errors="replace"))
        if res.returncode == 0
        else ""
    )
    gap_cer = compute_cer(gt_text, gap_text)
    gap_wer = compute_wer(gt_text, gap_text)

    results["gap_heuristic_psm_3"] = {
        "cer": round(gap_cer, 4),
        "wer": round(gap_wer, 4),
        "sample_output": [
            line.strip() for line in gap_text.splitlines() if line.strip()
        ][:4],
    }

    return results


# --- Timing Benchmark ---


def benchmark_timing(
    fixture_path: Path, engine: TesseractEngine, repetitions: int = 5
) -> dict:
    """Benchmark per-page extraction latency: native vs 200 DPI vs 300 DPI."""
    doc = pymupdf.open(str(fixture_path))
    page = doc[0]

    # Warmup
    _ = page.get_text("text")
    pix_warm = page.get_pixmap(dpi=200)
    _ = engine.recognize(pix_warm.tobytes("png"), language="eng")

    # 1. Native text extraction
    native_times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        _ = page.get_text("text")
        native_times.append((time.perf_counter() - t0) * 1000)

    # 2. 200 DPI OCR
    render_200_times = []
    engine_200_times = []
    total_200_times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        pix = page.get_pixmap(dpi=200)
        png = pix.tobytes("png")
        t_render = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        _ = engine.recognize(png, language="eng")
        t_engine = (time.perf_counter() - t1) * 1000

        render_200_times.append(t_render)
        engine_200_times.append(t_engine)
        total_200_times.append(t_render + t_engine)

    # 3. 300 DPI OCR
    render_300_times = []
    engine_300_times = []
    total_300_times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        pix = page.get_pixmap(dpi=300)
        png = pix.tobytes("png")
        t_render = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        _ = engine.recognize(png, language="eng")
        t_engine = (time.perf_counter() - t1) * 1000

        render_300_times.append(t_render)
        engine_300_times.append(t_engine)
        total_300_times.append(t_render + t_engine)

    doc.close()

    def _stats(arr: list[float]) -> dict[str, float]:
        return {
            "mean_ms": round(statistics.mean(arr), 2),
            "median_ms": round(statistics.median(arr), 2),
            "min_ms": round(min(arr), 2),
            "max_ms": round(max(arr), 2),
        }

    return {
        "repetitions": repetitions,
        "native": _stats(native_times),
        "ocr_200_dpi": {
            "render": _stats(render_200_times),
            "engine": _stats(engine_200_times),
            "total": _stats(total_200_times),
        },
        "ocr_300_dpi": {
            "render": _stats(render_300_times),
            "engine": _stats(engine_300_times),
            "total": _stats(total_300_times),
        },
    }


# --- Main Evaluation Runner ---


def run_evaluation(
    dpi_list: list[int] | None = None,
    fixtures_dir: Path = FIXTURES_DIR,
    ground_truth_dir: Path = GROUND_TRUTH_DIR,
) -> dict:
    """Run full evaluation suite across fixtures, timing, and tables."""
    if dpi_list is None:
        dpi_list = [200, 300]

    engine = TesseractEngine()
    try:
        tess_version = engine.version
        _ = engine._resolve_executable()
    except Exception as exc:
        print(
            "ERROR: Tesseract OCR is not installed or not available.\n"
            f"Details: {exc}\n"
            "Live evaluation requires Tesseract 5 (UB-Mannheim build).\n"
            "Aborting evaluation without generating fabricated results."
        )
        raise SystemExit(1) from exc
    pymupdf_version = getattr(pymupdf, "__version__", "unknown")
    machine_info = f"{platform.system()} {platform.release()} ({platform.machine()})"

    print("Starting Phase 10 OCR Evaluation")
    print(f"Tesseract Version: {tess_version}")
    print(f"PyMuPDF Version:   {pymupdf_version}")
    print(f"Platform:          {machine_info}")
    print("-" * 60)

    # Fixture List
    fixture_specs = [
        ("clean_single", "clean_single.pdf", "clean_single.txt"),
        ("multi_paragraph", "multi_paragraph.pdf", "multi_paragraph.txt"),
        ("headings_punct", "headings_punct.pdf", "headings_punct.txt"),
        ("scan_pdf_3page", "scan_pdf_3page.pdf", "scan_pdf_3page.txt"),
        ("mixed_pdf", "mixed_pdf.pdf", "mixed_pdf.txt"),
        ("image_png", "image_png.png", "image_png.txt"),
        ("image_jpg", "image_jpg.jpg", "image_jpg.txt"),
        ("degraded_jpg", "degraded_jpg.jpg", "degraded_jpg.txt"),
        ("table_simple", "table_simple.pdf", "table_simple.txt"),
        ("table_borderless", "table_borderless.pdf", "table_borderless.txt"),
    ]

    fixture_results: dict[int, dict] = {}

    for dpi in dpi_list:
        print(f"\nEvaluating Fixtures at DPI = {dpi}...")
        processor = OcrProcessor(engine=engine, dpi=dpi)
        dpi_evals = {}
        cers = []
        wers = []

        for fix_id, fix_fname, gt_fname in fixture_specs:
            fix_path = fixtures_dir / fix_fname
            gt_path = ground_truth_dir / gt_fname

            if not fix_path.exists() or not gt_path.exists():
                print(f"  [MISSING] {fix_fname} or {gt_fname}")
                continue

            gt_text = gt_path.read_text(encoding="utf-8")
            t0 = time.perf_counter()
            hyp_text, pages = extract_document_text(fix_path, processor)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            cer = compute_cer(gt_text, hyp_text)
            wer = compute_wer(gt_text, hyp_text)
            cers.append(cer)
            wers.append(wer)

            page_cers = []
            if len(pages) > 1:
                # Per page CER if multi-page
                gt_pages = [p for p in gt_text.split("\n\n") if p.strip()]
                for idx, p_text in enumerate(pages):
                    ref_p = gt_pages[idx] if idx < len(gt_pages) else ""
                    page_cers.append(round(compute_cer(ref_p, p_text), 4))

            dpi_evals[fix_id] = {
                "file": fix_fname,
                "cer": round(cer, 4),
                "wer": round(wer, 4),
                "page_cers": page_cers if page_cers else None,
                "latency_ms": round(elapsed_ms, 2),
                "num_pages": len(pages),
                "char_count_ref": len(gt_text),
                "char_count_hyp": len(hyp_text),
            }
            print(
                f"  {fix_id:<18} | CER: {cer:.4f} | WER: {wer:.4f} | "
                f"Latency: {elapsed_ms:>6.1f}ms"
            )

        macro_cer = statistics.mean(cers) if cers else 0.0
        macro_wer = statistics.mean(wers) if wers else 0.0
        fixture_results[dpi] = {
            "fixtures": dpi_evals,
            "macro_cer": round(macro_cer, 4),
            "macro_wer": round(macro_wer, 4),
        }

    # Timing comparison (using clean_single)
    print("\nRunning Timing Benchmark (Native vs 200 DPI vs 300 DPI)...")
    timing_benchmark = benchmark_timing(
        fixtures_dir / "clean_single.pdf", engine, repetitions=5
    )

    # Table assessment
    print("\nRunning Table Layout Assessment...")
    table_simple_eval = evaluate_table_modes(
        fixtures_dir / "table_simple.pdf",
        ground_truth_dir / "table_simple.txt",
        engine,
    )
    table_borderless_eval = evaluate_table_modes(
        fixtures_dir / "table_borderless.pdf",
        ground_truth_dir / "table_borderless.txt",
        engine,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "metadata": {
            "timestamp": timestamp,
            "engine": "tesseract",
            "tesseract_version": tess_version,
            "pymupdf_version": pymupdf_version,
            "platform": machine_info,
            "python_version": platform.python_version(),
        },
        "dpi_evaluations": fixture_results,
        "timing_benchmark": timing_benchmark,
        "table_assessment": {
            "table_simple": table_simple_eval,
            "table_borderless": table_borderless_eval,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"ocr_eval_{timestamp}.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved structured evaluation report to {out_file}")

    _print_markdown_summary(report)
    return report


def _print_markdown_summary(report: dict) -> None:
    """Print markdown-formatted summary for docs/ocr.md."""
    meta = report["metadata"]
    dpi_evals = report["dpi_evaluations"]
    timing = report["timing_benchmark"]

    print("\n" + "=" * 60)
    print("### Phase 10 Measured OCR Evaluation Summary")
    print("=" * 60)
    print(f"- **Engine**: Tesseract `{meta['tesseract_version']}`")
    print(
        f"- **Environment**: `{meta['platform']}` / Python `{meta['python_version']}`"
    )
    print(f"- **Date**: {meta['timestamp']}")
    print()

    print("#### Quality Metrics (CER & WER across Fixtures)")
    print(
        "| Fixture | 200 DPI CER | 200 DPI WER | 300 DPI CER | 300 DPI WER "
        "| Description |"
    )
    print("|---|---|---|---|---|---|")

    f200 = dpi_evals.get(200, {}).get("fixtures", {})
    f300 = dpi_evals.get(300, {}).get("fixtures", {})

    descriptions = {
        "clean_single": "Clean 12pt single paragraph (baseline)",
        "multi_paragraph": "3 paragraphs reading order",
        "headings_punct": "Headings, punctuation, numbers",
        "scan_pdf_3page": "3-page image-only PDF",
        "mixed_pdf": "Mixed native + scanned pages",
        "image_png": "Direct PNG image",
        "image_jpg": "Direct JPEG image (q=75)",
        "degraded_jpg": "Degraded JPEG (150 DPI, q=35)",
        "table_simple": "Ruled 4x3 table",
        "table_borderless": "Borderless 4x3 table",
    }

    for fix_id, desc in descriptions.items():
        res200 = f200.get(fix_id, {})
        res300 = f300.get(fix_id, {})
        c200 = res200.get("cer", "N/A")
        w200 = res200.get("wer", "N/A")
        c300 = res300.get("cer", "N/A")
        w300 = res300.get("wer", "N/A")
        print(f"| `{fix_id}` | {c200} | {w200} | {c300} | {w300} | {desc} |")

    m200_c = dpi_evals.get(200, {}).get("macro_cer", "N/A")
    m200_w = dpi_evals.get(200, {}).get("macro_wer", "N/A")
    m300_c = dpi_evals.get(300, {}).get("macro_cer", "N/A")
    m300_w = dpi_evals.get(300, {}).get("macro_wer", "N/A")
    print(
        f"| **Macro Average** | **{m200_c}** | **{m200_w}** | **{m300_c}** | "
        f"**{m300_w}** | All fixtures |"
    )

    print("\n#### Per-Page Latency Benchmark (clean_single, 5 trials)")
    print(
        "| Method | Render Mean (ms) | Engine Mean (ms) | Total Mean (ms) | "
        "Total Median (ms) |"
    )
    print("|---|---|---|---|---|")
    nat = timing["native"]
    o200 = timing["ocr_200_dpi"]
    o300 = timing["ocr_300_dpi"]
    print(f"| Native text | — | — | {nat['mean_ms']} | {nat['median_ms']} |")
    print(
        f"| OCR 200 DPI | {o200['render']['mean_ms']} | {o200['engine']['mean_ms']} | "
        f"{o200['total']['mean_ms']} | {o200['total']['median_ms']} |"
    )
    print(
        f"| OCR 300 DPI | {o300['render']['mean_ms']} | {o300['engine']['mean_ms']} | "
        f"{o300['total']['mean_ms']} | {o300['total']['median_ms']} |"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 10 OCR Evaluation")
    parser.add_argument(
        "--dpi",
        nargs="+",
        type=int,
        default=[200, 300],
        help="DPI resolutions to evaluate (default: 200 300)",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=FIXTURES_DIR,
        help="Directory containing test fixtures",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=GROUND_TRUTH_DIR,
        help="Directory containing ground truth text files",
    )
    args = parser.parse_args()

    run_evaluation(
        dpi_list=args.dpi,
        fixtures_dir=args.fixtures_dir,
        ground_truth_dir=args.ground_truth_dir,
    )
