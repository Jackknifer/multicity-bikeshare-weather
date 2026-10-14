import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import WORK_ROOT, load_paths, read_json, verify_frozen_input

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append({"check": name, "passed": bool(condition), "detail": str(detail)})
    return bool(condition)


def run(command):
    return subprocess.run(command, capture_output=True, text=True)


def pdf_fonts_not_embedded(path):
    result = run(["pdffonts", str(path)])
    if result.returncode != 0:
        raise RuntimeError(f"pdffonts failed for {path}: {result.stderr}")
    lines = [line for line in result.stdout.splitlines()[2:] if line.strip()]
    missing = []
    for line in lines:
        fields = line.split()
        if len(fields) < 6:
            raise ValueError(f"unexpected pdffonts row: {line}")
        if fields[-5] == "no":
            missing.append(line)
    return missing


def pdf_pages(path):
    result = run(["pdfinfo", str(path)])
    if result.returncode != 0:
        raise RuntimeError(f"pdfinfo failed for {path}: {result.stderr}")
    match = re.search(r"Pages:\s+(\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"cannot read page count for {path}")
    return int(match.group(1))


def pdf_text(path, first=None, last=None):
    command = ["pdftotext", "-layout"]
    if first is not None:
        command += ["-f", str(first)]
    if last is not None:
        command += ["-l", str(last)]
    command += [str(path), "-"]
    result = run(command)
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed for {path}")
    return result.stdout


def references_start_page(path):
    total = pdf_pages(path)
    for page in range(1, total + 1):
        text = pdf_text(path, page, page)
        for line in text.splitlines():
            if re.match(r"^\s*References\s*$", line):
                return page
    return None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="formal_20260930")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    out = Path(args.out) if args.out else WORK_ROOT / "docs" / "deliverable-verification.json"

    for relative in (
        "configs/paths.github.yaml", "configs/splits.json", "configs/models.json",
        "data/city_data_audit.csv", "data/split_index.csv", "data/split_summary.json",
        "data/integrity_report.json",
        "docs/course_requirements.md", "docs/data_audit.md", "docs/analysis_protocol.md",
        "tables/all_metrics.csv", "tables/selection.json",
        "report/final_report.pdf", "report/final_report.tex",
        "slides/final_slides.pdf", "slides/final_slides.tex",
        "README.md", "requirements.txt",
    ):
        check(f"exists {relative}", (WORK_ROOT / relative).exists())

    paths = load_paths()
    integrity = verify_frozen_input(paths)
    check("frozen data hash verified", integrity["sha256"] == paths["data_sha256"], integrity["sha256"])

    selection = read_json(WORK_ROOT / "tables" / "selection.json")
    selected = selection["selected_model"]
    check("selected model recorded", bool(selected), selected)

    metrics = pd.read_csv(WORK_ROOT / "tables" / "all_metrics.csv")
    formal = metrics.loc[metrics["model"].str.startswith(("M", "A"))]
    check("all formal fits pass diagnostics", bool(formal["diagnostics_passed"].all()),
          f"{int((~formal['diagnostics_passed']).sum())} failing rows")
    check("no divergences in formal fits", bool((formal["divergences"] == 0).all()))
    check("Rhat threshold met", bool((formal["max_rhat"] <= 1.01).all()),
          f"max {formal['max_rhat'].max():.5f}")
    check("bulk ESS threshold met", bool((formal["min_ess_bulk"] >= 400).all()),
          f"min {formal['min_ess_bulk'].min():.1f}")
    check("tail ESS threshold met", bool((formal["min_ess_tail"] >= 400).all()),
          f"min {formal['min_ess_tail'].min():.1f}")

    block = metrics.loc[metrics["task"] == "block"]
    dev = block.loc[block["fold"].isin(["dev1", "dev2"]) & block["model"].str.startswith("M")]
    for fold in ("dev1", "dev2"):
        rows = dev.loc[dev["fold"] == fold]
        check(f"{fold} has M0, M1 and M2", set(rows["model"]) >= {"M0", "M1", "M2"},
              sorted(rows["model"]))
    final = block.loc[block["fold"] == "final"]
    check("final fold has the reference and the selected model",
          {"M0", selected} <= set(final["model"]), sorted(set(final["model"])))

    for model in ("M0", "M1", "M2"):
        for fold in ("dev1", "dev2"):
            directory = WORK_ROOT / "runs" / args.run_id / model / fold
            for relative in ("posterior.nc", "predictions.csv", "metrics_city.csv",
                             "diagnostics.json", "config.json", "preprocessing.json",
                             "model_spec.md"):
                check(f"{model}/{fold} has {relative}", (directory / relative).exists())

    predictions = pd.read_csv(
        WORK_ROOT / "runs" / args.run_id / "M0" / "dev1" / "predictions.csv", parse_dates=["date"]
    )
    check("block predictions are 56 days", int(predictions.loc[predictions["task"] == "block", "horizon"].max()) == 56)
    check("both prediction tasks saved", set(predictions["task"]) == {"block", "one_step"})

    report = WORK_ROOT / "report" / "final_report.pdf"
    if report.exists():
        total = pdf_pages(report)
        start = references_start_page(report)
        check("report references section found", start is not None, f"total pages {total}")
        if start:
            reference_page = pdf_text(report, start, start)
            preceding_text = reference_page.split("References", 1)[0].strip()
            main_pages = start if preceding_text else start - 1
            check("main text within nine pages", main_pages <= 9,
                  f"main text pages {main_pages}, total {total}")
        fonts = run(["pdffonts", str(report)]).stdout
        not_embedded = pdf_fonts_not_embedded(report)
        check("report fonts embedded", not not_embedded, not_embedded[:3])
        del fonts
        text = pdf_text(report)
        check("report has searchable text", len(text.split()) > 1500, f"{len(text.split())} words")
        contribution = ""
        if "Contribution statement" in text:
            contribution = text.split("Contribution statement", 1)[1].split("References", 1)[0]
        check("contribution statement filled",
              all(name in contribution for name in ("XUE Yantong", "WANG Ziyu", "ZHAO Jiashen"))
              and len(contribution.split()) > 30)

    slides = WORK_ROOT / "slides" / "final_slides.pdf"
    if slides.exists():
        pages = pdf_pages(slides)
        check("slides compiled", pages >= 8, f"{pages} pages")
        check("slide fonts embedded", not pdf_fonts_not_embedded(slides))

    figure_dir = WORK_ROOT / "figures"
    figures = sorted(figure_dir.glob("*.pdf"))
    check("at least seven figures produced", len(figures) >= 7, len(figures))
    for figure in figures:
        png = figure.with_suffix(".png")
        if not check(f"{figure.name} has PNG", png.exists()):
            continue
        with Image.open(png) as image:
            width, height = image.size
            dpi = image.info.get("dpi")
        check(f"{figure.stem} PNG is 300 dpi",
              dpi is not None and all(abs(value - 300.0) < 1.0 for value in dpi),
              f"{width}x{height}, dpi={dpi}")

    failures = [item for item in RESULTS if not item["passed"]]
    summary = {
        "run_id": args.run_id,
        "total": len(RESULTS),
        "passed": len(RESULTS) - len(failures),
        "failed": len(failures),
        "failures": failures,
        "checks": RESULTS,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for item in RESULTS:
        print(f"{'PASS' if item['passed'] else 'FAIL'} {item['check']} {item['detail']}")
    print(f"\n{summary['passed']}/{summary['total']} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
