import os
import json
import csv
from pathlib import Path
import numpy as np

try:
    from scipy.stats import ttest_ind
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =========================================================
# PATH HELPERS
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUN_ROOT = PROJECT_ROOT / "03_runs"

# Optional:
# - set RUN_DIR_NAME to a specific folder name under 03_runs
# - leave as None to auto-pick the latest run folder containing records.json
RUN_DIR_NAME = None

# Example:
# RUN_DIR_NAME = "pilot_montecarlo_v1_COMPARE_20260421_200317"


def find_latest_run_dir(run_root: Path) -> Path:
    candidates = [
        p for p in run_root.iterdir()
        if p.is_dir() and (p / "records.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No run folders with records.json found in: {run_root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


if RUN_DIR_NAME:
    RUN_DIR = RUN_ROOT / RUN_DIR_NAME
else:
    RUN_DIR = find_latest_run_dir(RUN_ROOT)

RECORDS_PATH = RUN_DIR / "records.json"

if not RECORDS_PATH.exists():
    raise FileNotFoundError(f"records.json not found: {RECORDS_PATH}")

# =========================================================
# ANALYSIS OUTPUT DIRECTORY
# =========================================================
ANALYSIS_DIR = RUN_DIR / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================
with open(RECORDS_PATH, "r", encoding="utf-8") as f:
    records = json.load(f)

if not records:
    raise ValueError("records.json is empty.")

ARCHS = sorted({r["arch"] for r in records})
TEMPS = sorted({r["temperature"] for r in records})


# =========================================================
# UTILS
# =========================================================
def clean_vals(vals):
    return [v for v in vals if v is not None]


def summarize(vals):
    vals = clean_vals(vals)
    if not vals:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "min": None,
            "max": None
        }
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return {
        "n": len(vals),
        "mean": float(np.mean(vals)),
        "sd": sd,
        "min": float(np.min(vals)),
        "max": float(np.max(vals))
    }


def icc_1k(matrix):
    """
    ICC(1,k) for ratings matrix:
    rows = targets/items (answer_id)
    cols = repeated runs (temperature x repeat)
    """
    X = np.array(matrix, dtype=float)
    if X.ndim != 2:
        return None

    n, k = X.shape
    if n < 2 or k < 2:
        return None

    row_means = X.mean(axis=1, keepdims=True)
    grand_mean = X.mean()

    SSR = k * np.sum((row_means - grand_mean) ** 2)
    SSE = np.sum((X - row_means) ** 2)

    MSR = SSR / (n - 1)
    MSE = SSE / (n * (k - 1))

    denom = MSR + (k - 1) * MSE
    if denom == 0:
        return None

    return float((MSR - MSE) / denom)


def cohen_d_independent(x, y):
    x = clean_vals(x)
    y = clean_vals(y)
    if len(x) < 2 or len(y) < 2:
        return None

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    nx = len(x)
    ny = len(y)
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)

    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_sd)


def welch_t_test(x, y):
    x = clean_vals(x)
    y = clean_vals(y)
    if len(x) < 2 or len(y) < 2 or not SCIPY_AVAILABLE:
        return None, None

    result = ttest_ind(x, y, equal_var=False, nan_policy="omit")
    return float(result.statistic), float(result.pvalue)


def build_icc_matrix(subset, score_key="total_score"):
    """
    Build matrix:
    - rows = answer_id
    - cols = sorted (temperature, repeat) combinations
    """
    answer_ids = sorted({r["answer_id"] for r in subset})
    run_keys = sorted({(r["temperature"], r["repeat"]) for r in subset})

    matrix = []
    for aid in answer_ids:
        row = []
        for temp, rep in run_keys:
            match = next(
                (
                    r for r in subset
                    if r["answer_id"] == aid
                    and r["temperature"] == temp
                    and r["repeat"] == rep
                ),
                None
            )
            row.append(None if match is None else match.get(score_key))

        if all(v is not None for v in row):
            matrix.append(row)

    return matrix


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x, digits=6):
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


# =========================================================
# SUMMARY TABLES
# =========================================================
metric_map = {
    "total_score": "Total Score",
    "grounding_rate": "Grounding Rate",
    "mean_coherence": "Mean Coherence",
    "contradiction_rate": "Contradiction Rate",
}

overall_rows = []
by_temp_rows = []
icc_rows = []
comparison_rows = []

# Overall summary by architecture
for arch in ARCHS:
    subset_arch = [r for r in records if r["arch"] == arch]

    for metric_key, metric_label in metric_map.items():
        vals = [r.get(metric_key) for r in subset_arch]
        stats = summarize(vals)
        overall_rows.append({
            "arch": arch,
            "metric": metric_key,
            "metric_label": metric_label,
            **stats
        })

    matrix = build_icc_matrix(subset_arch, score_key="total_score")
    icc_val = icc_1k(matrix) if len(matrix) >= 2 else None
    icc_rows.append({
        "arch": arch,
        "metric": "total_score",
        "icc_1k": icc_val,
        "n_answers_used": len(matrix),
        "n_repeated_runs_per_answer": len(matrix[0]) if matrix else 0
    })

# Summary by architecture × temperature
for arch in ARCHS:
    for temp in TEMPS:
        subset = [r for r in records if r["arch"] == arch and r["temperature"] == temp]

        for metric_key, metric_label in metric_map.items():
            vals = [r.get(metric_key) for r in subset]
            stats = summarize(vals)
            by_temp_rows.append({
                "arch": arch,
                "temperature": temp,
                "metric": metric_key,
                "metric_label": metric_label,
                **stats
            })

# Overall architecture comparison
if "single" in ARCHS and "multi" in ARCHS:
    single_subset = [r for r in records if r["arch"] == "single"]
    multi_subset = [r for r in records if r["arch"] == "multi"]

    for metric_key, metric_label in metric_map.items():
        x = [r.get(metric_key) for r in single_subset]
        y = [r.get(metric_key) for r in multi_subset]

        x_stats = summarize(x)
        y_stats = summarize(y)
        t_stat, p_val = welch_t_test(x, y)
        d_val = cohen_d_independent(x, y)

        comparison_rows.append({
            "metric": metric_key,
            "metric_label": metric_label,
            "single_n": x_stats["n"],
            "single_mean": x_stats["mean"],
            "single_sd": x_stats["sd"],
            "multi_n": y_stats["n"],
            "multi_mean": y_stats["mean"],
            "multi_sd": y_stats["sd"],
            "welch_t": t_stat,
            "p_value": p_val,
            "cohens_d": d_val
        })


# =========================================================
# SAVE OUTPUTS
# =========================================================
overall_csv = ANALYSIS_DIR / "analysis_summary_by_architecture.csv"
by_temp_csv = ANALYSIS_DIR / "analysis_summary_by_architecture_and_temperature.csv"
icc_csv = ANALYSIS_DIR / "analysis_icc_by_architecture.csv"
comparison_csv = ANALYSIS_DIR / "analysis_architecture_comparison.csv"
json_out = ANALYSIS_DIR / "analysis_summary_full.json"
txt_out = ANALYSIS_DIR / "analysis_report.txt"

write_csv(
    overall_csv,
    overall_rows,
    ["arch", "metric", "metric_label", "n", "mean", "sd", "min", "max"]
)

write_csv(
    by_temp_csv,
    by_temp_rows,
    ["arch", "temperature", "metric", "metric_label", "n", "mean", "sd", "min", "max"]
)

write_csv(
    icc_csv,
    icc_rows,
    ["arch", "metric", "icc_1k", "n_answers_used", "n_repeated_runs_per_answer"]
)

if comparison_rows:
    write_csv(
        comparison_csv,
        comparison_rows,
        [
            "metric", "metric_label",
            "single_n", "single_mean", "single_sd",
            "multi_n", "multi_mean", "multi_sd",
            "welch_t", "p_value", "cohens_d"
        ]
    )

with open(json_out, "w", encoding="utf-8") as f:
    json.dump(
        {
            "run_dir": str(RUN_DIR),
            "analysis_dir": str(ANALYSIS_DIR),
            "records_path": str(RECORDS_PATH),
            "overall_by_architecture": overall_rows,
            "by_architecture_and_temperature": by_temp_rows,
            "icc_by_architecture": icc_rows,
            "architecture_comparison": comparison_rows,
        },
        f,
        indent=2,
        ensure_ascii=False
    )


# =========================================================
# HUMAN-READABLE REPORT
# =========================================================
lines = []
lines.append("ANALYSIS REPORT")
lines.append(f"Run directory: {RUN_DIR}")
lines.append(f"Analysis dir : {ANALYSIS_DIR}")
lines.append(f"Records file : {RECORDS_PATH}")
lines.append("")

lines.append("1) OVERALL SUMMARY BY ARCHITECTURE")
for row in overall_rows:
    lines.append(
        f"- {row['arch']} | {row['metric_label']}: "
        f"n={row['n']}, mean={fmt(row['mean'])}, sd={fmt(row['sd'])}, "
        f"min={fmt(row['min'])}, max={fmt(row['max'])}"
    )

lines.append("")
lines.append("2) ICC(1,k) TOTAL SCORE BY ARCHITECTURE")
for row in icc_rows:
    lines.append(
        f"- {row['arch']}: ICC(1,k)={fmt(row['icc_1k'])}, "
        f"answers_used={row['n_answers_used']}, "
        f"repeated_runs_per_answer={row['n_repeated_runs_per_answer']}"
    )

if comparison_rows:
    lines.append("")
    lines.append("3) OVERALL ARCHITECTURE COMPARISON")
    for row in comparison_rows:
        lines.append(
            f"- {row['metric_label']}: "
            f"single_mean={fmt(row['single_mean'])}, "
            f"multi_mean={fmt(row['multi_mean'])}, "
            f"welch_t={fmt(row['welch_t'])}, "
            f"p_value={fmt(row['p_value'])}, "
            f"cohens_d={fmt(row['cohens_d'])}"
        )

lines.append("")
lines.append("4) OUTPUT FILES")
lines.append(f"- {overall_csv.name}")
lines.append(f"- {by_temp_csv.name}")
lines.append(f"- {icc_csv.name}")
if comparison_rows:
    lines.append(f"- {comparison_csv.name}")
lines.append(f"- {json_out.name}")
lines.append(f"- {txt_out.name}")

with open(txt_out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))


# =========================================================
# CONSOLE OUTPUT
# =========================================================
print(f"Analyzed run folder: {RUN_DIR}")
print(f"Analysis output dir: {ANALYSIS_DIR}")
print("Saved output files:")
print(f"- {overall_csv}")
print(f"- {by_temp_csv}")
print(f"- {icc_csv}")
if comparison_rows:
    print(f"- {comparison_csv}")
print(f"- {json_out}")
print(f"- {txt_out}")

print("\nICC SUMMARY:")
for row in icc_rows:
    print(
        f"  {row['arch']}: ICC(1,k)={fmt(row['icc_1k'])}, "
        f"answers_used={row['n_answers_used']}, "
        f"repeated_runs_per_answer={row['n_repeated_runs_per_answer']}"
    )