"""
Final AUC Plots v2
------------------
Generates model-comparison and panel-comparison ROC figures for each
classification task, using SEPARATE per-cohort curves rather than
pooling all cohort predictions together.

FIXED (v1 bug): v1 concatenated all cohorts before calling roc_curve(),
which produced a single pooled AUC that matched neither individual cohort
values nor the cross-cohort mean reported in the tables.

v2 behaviour:
  - Internal LOCO:  one pooled prediction set → single curve (correct,
    because LOCO predictions are already one-fold-at-a-time aggregates)
  - External:       one curve per cohort + one mean-AUC annotation
    (multi-cohort tasks: ATB vs LTB, ATB vs OD have 2 cohorts each;
     LTB vs CON has 1 cohort)

AUC labels in legends now match the values reported in Tables 1–6.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc, roc_auc_score

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

PANEL_COLORS = {
    "Native20":  "#3B82F6",
    "Sweeney3":  "#F0E442",
    "Zak16":     "#D55E00",
    "Kaforou":   "#CC79A7",
    "Berry86":   "#009E73",
    "Random20":  "#000000",
}

MODEL_COLORS = {
    "LOGREG": "#E15759",
    "MLP":    "#4E79A7",
    "RF":     "#59A14F",
}

MODEL_LABELS = {
    "LOGREG": "Logistic Regression",
    "MLP":    "MLP",
    "RF":     "Random Forest",
}

plt.rcParams.update({
    "font.family":     "Arial",
    "font.size":       12,
    "axes.labelsize":  14,
    "axes.titlesize":  14,
    "legend.fontsize": 10,
})

# ---------------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------------

def _norm_model(s: str) -> str:
    s = str(s).strip().upper()
    if s in ("LOGREG", "LOGISTICREGRESSION", "LR"):
        return "LOGREG"
    if s in ("MLP", "MLPCLASSIFIER"):
        return "MLP"
    if s in ("RF", "RANDOMFOREST", "RANDOMFORESTCLASSIFIER"):
        return "RF"
    return s

def _norm_panel(s: str) -> str:
    s = str(s).strip().lower().replace(" ", "")
    if "native" in s:
        return "Native20"
    if "zak" in s:
        return "Zak16"
    if "berry" in s:
        return "Berry86"
    if "sweeney" in s:
        return "Sweeney3"
    if "kaforou" in s:
        return "Kaforou"
    if "random" in s:
        return "Random20"
    return s

def _norm_cohort(s: str) -> str:
    """Strip suffix noise, keep GSExxxxxx."""
    s = str(s)
    for suffix in ("_zscored_final", "_zscored", "_final"):
        s = s.replace(suffix, "")
    return s.strip()

def load_predictions(root: Path) -> pd.DataFrame:
    records = []
    for f in root.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in (".csv", ".tsv", ".txt"):
            continue
        for sep in (",", "\t"):
            try:
                df = pd.read_csv(f, sep=sep)
            except Exception:
                continue
            required = {"TrueLabel", "PredProb", "Task", "Model", "Panel"}
            if not required.issubset(df.columns):
                continue
            df = df.copy()
            parts_upper = [p.upper() for p in f.parts]
            evaluation = "External" if any("EXTERNAL" in p for p in parts_upper) else "Internal"
            df["Evaluation"] = evaluation

            if "Cohort" in df.columns:
                df["Cohort"] = df["Cohort"].apply(_norm_cohort)
            else:
                stem = f.stem
                import re
                m = re.search(r"(GSE\d+)", stem)
                df["Cohort"] = m.group(1) if m else ("LOCO" if evaluation == "Internal" else "Unknown")

            df["Model"] = df["Model"].apply(_norm_model)
            df["Panel"] = df["Panel"].apply(_norm_panel)

            records.append(df[["TrueLabel", "PredProb", "Task", "Model", "Panel", "Cohort", "Evaluation"]])
            break

    if not records:
        raise RuntimeError(f"No prediction files found under {root}")

    combined = pd.concat(records, ignore_index=True)
    print(f"Loaded {len(combined):,} predictions from {len(records)} files")
    print("  Tasks:", sorted(combined.Task.unique()))
    print("  Models:", sorted(combined.Model.unique()))
    print("  Panels:", sorted(combined.Panel.unique()))
    print("  Cohorts:", sorted(combined.Cohort.unique()))
    print("  Evaluations:", sorted(combined.Evaluation.unique()))
    return combined

# ---------------------------------------------------------------------------
# PLOTTING HELPERS
# ---------------------------------------------------------------------------

def _plot_base(ax):
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path.name}")

# ---------------------------------------------------------------------------
# FIGURE A: Model comparison (panel fixed = Native20)
# ---------------------------------------------------------------------------

def plot_model_comparison(df: pd.DataFrame, task: str, evaluation: str, outdir: Path):
    sub = df[
        (df["Task"] == task) &
        (df["Evaluation"] == evaluation) &
        (df["Panel"] == "Native20")
    ]
    if sub.empty:
        warnings.warn(f"No data: model comparison / {task} / {evaluation}")
        return

    safe_task = task.replace(" ", "_")

    if evaluation == "Internal":
        fig, ax = plt.subplots(figsize=(6, 6))
        _plot_base(ax)
        for model in sorted(sub["Model"].unique()):
            msub = sub[sub["Model"] == model]
            if msub["TrueLabel"].nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(msub["TrueLabel"], msub["PredProb"])
            auc_val = sk_auc(fpr, tpr)
            color = MODEL_COLORS.get(model, "#333333")
            label = f"{MODEL_LABELS.get(model, model)} (AUC = {auc_val:.3f})"
            ax.plot(fpr, tpr, color=color, linewidth=2, label=label)

        ax.set_title(f"{task} — Models (Internal LOCO, Native20)")
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout()
        _save(fig, outdir / f"{safe_task}_Internal_models_ROC.tiff")

    else:
        cohorts = sorted(sub["Cohort"].unique())
        n = len(cohorts)

        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)

        for ci, cohort in enumerate(cohorts):
            ax = axes[0][ci]
            _plot_base(ax)

            for model in sorted(sub["Model"].unique()):
                csub = sub[(sub["Model"] == model) & (sub["Cohort"] == cohort)]

                if csub.empty or csub["TrueLabel"].nunique() < 2:
                    continue

                fpr, tpr, _ = roc_curve(csub["TrueLabel"], csub["PredProb"])
                auc_val = sk_auc(fpr, tpr)

                color = MODEL_COLORS.get(model, "#333333")

                ax.plot(
                    fpr,
                    tpr,
                    color=color,
                    linewidth=2,
                    label=f"{MODEL_LABELS.get(model, model)} (AUC = {auc_val:.3f})",
                )

            ax.set_title(f"{task} — Models\n{cohort}")
            ax.legend(loc="lower right", frameon=False, fontsize=9)

        fig.tight_layout()
        _save(fig, outdir / f"{safe_task}_External_models_ROC.tiff")

# ---------------------------------------------------------------------------
# FIGURE B: Panel comparison (model fixed = MLP)
# ---------------------------------------------------------------------------

def plot_panel_comparison(df: pd.DataFrame, task: str, evaluation: str, outdir: Path):
    sub = df[
        (df["Task"] == task) &
        (df["Evaluation"] == evaluation) &
        (df["Model"] == "MLP")
    ]
    if sub.empty:
        warnings.warn(f"No data: panel comparison / {task} / {evaluation}")
        return

    safe_task = task.replace(" ", "_")

    if evaluation == "Internal":
        fig, ax = plt.subplots(figsize=(6, 6))
        _plot_base(ax)

        for panel in sorted(sub["Panel"].unique()):
            psub = sub[sub["Panel"] == panel]

            if psub["TrueLabel"].nunique() < 2:
                continue

            fpr, tpr, _ = roc_curve(psub["TrueLabel"], psub["PredProb"])
            auc_val = sk_auc(fpr, tpr)

            color = PANEL_COLORS.get(panel, "#555555")

            ax.plot(
                fpr,
                tpr,
                color=color,
                linewidth=2,
                label=f"{panel} (AUC = {auc_val:.3f})",
            )

        ax.set_title(f"{task} — Panels (Internal LOCO, MLP)")
        ax.legend(loc="lower right", frameon=False, fontsize=9)

        fig.tight_layout()
        _save(fig, outdir / f"{safe_task}_Internal_panels_ROC.tiff")

    else:
        cohorts = sorted(sub["Cohort"].unique())
        n = len(cohorts)

        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)

        for ci, cohort in enumerate(cohorts):
            ax = axes[0][ci]
            _plot_base(ax)

            for panel in sorted(sub["Panel"].unique()):
                csub = sub[(sub["Panel"] == panel) & (sub["Cohort"] == cohort)]

                if csub.empty or csub["TrueLabel"].nunique() < 2:
                    continue

                fpr, tpr, _ = roc_curve(csub["TrueLabel"], csub["PredProb"])
                auc_val = sk_auc(fpr, tpr)

                color = PANEL_COLORS.get(panel, "#555555")

                ax.plot(
                    fpr,
                    tpr,
                    color=color,
                    linewidth=2,
                    label=f"{panel} (AUC = {auc_val:.3f})",
                )

            ax.set_title(f"{task} — Panels\n{cohort}")
            ax.legend(loc="lower right", frameon=False, fontsize=9)

        fig.tight_layout()
        _save(fig, outdir / f"{safe_task}_External_panels_ROC.tiff")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    root = Path(r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Supplementary\Final Benchmarking results")

    outdir = Path(r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Supplementary\AUC Plots")

    outdir.mkdir(parents=True, exist_ok=True)

    df = load_predictions(root)

    tasks = sorted(df["Task"].unique())
    evaluations = ["Internal", "External"]

    for task in tasks:
        for evaluation in evaluations:
            print(f"\n--- {task} / {evaluation} ---")
            plot_model_comparison(df, task, evaluation, outdir)
            plot_panel_comparison(df, task, evaluation, outdir)

    print(f"\nDone. All figures written to: {outdir}")

if __name__ == "__main__":
    main()