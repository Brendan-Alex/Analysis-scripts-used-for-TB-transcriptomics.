"""
PCA Analysis v2
---------------
Runs all three classification tasks in one go.
For each task, generates:
  1. <task>_PCA_train.tiff          — training cohorts only
  2. <task>_PCA_<cohort>.tiff       — each external test cohort individually
  3. <task>_PCA_combined.tiff       — all sources + classes combined

PCA is always fit on training data only, then used to project everything.

Usage:
    python PCA_analysis_v2.py <data_root> <panel_root> <output_dir>

    data_root    folder containing Train Datasets Normalized/ and
                 Test Datasets Normalized/ subdirectories
    panel_root   folder containing the gene panel CSVs
    output_dir   where to save the .tiff files (created if absent)

Example:
    python PCA_analysis_v2.py \
        "C:/Users/brend/Desktop/Project Data/Combined final dataset/Test+Train Datasets" \
        "C:/Users/brend/Desktop/Project Data/Combined final dataset/Supplementary/Gene Panels" \
        "C:/Users/brend/Desktop/Project Data/Combined final dataset/Supplementary/PCA Plots"
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# TASK CONFIG
# ---------------------------------------------------------------------------

TASKS = {
    "ATB_vs_LTB": {
        "class1": "ATB", "class2": "LTB",
        "train_cohorts": ["GSE19491", "GSE37250", "GSE73408"],
        "test_cohorts":  ["GSE101705", "GSE107994"],
        "panel_file":    "Native 20 genes.csv",
    },
    "ATB_vs_OD": {
        "class1": "ATB", "class2": "OD",
        "train_cohorts": ["GSE19491", "GSE37250", "GSE73408"],
        "test_cohorts":  ["GSE42834", "GSE83456"],
        "panel_file":    "Native 20 genes.csv",
    },
    "LTB_vs_CON": {
        "class1": "LTB", "class2": "CON",
        "train_cohorts": ["GSE19439", "GSE19444", "GSE28623"],
        "test_cohorts":  ["GSE107994"],
        "panel_file":    "Native 20 genes.csv",
    },
}

plt.rcParams.update({
    "font.family":    "Arial",
    "font.size":      12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
})

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def normalize_gene(g):
    return str(g).strip().upper().split(".")[0]


def find_cohort_file(data_root: Path, cohort: str) -> Path:
    """Search train and test subdirs for a file whose stem starts with cohort."""
    for subdir in ("Train Datasets/Train Datasets Normalized",
                   "Test Datasets/Test Datasets Normalized"):
        folder = data_root / subdir
        matches = list(folder.glob(f"{cohort}*.csv"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find a CSV for cohort '{cohort}' under {data_root}"
    )


def load_cohort(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, engine="python")
    df.columns = df.columns.str.strip()
    df["Gene"] = df["Gene"].apply(normalize_gene)
    return df


def extract_matrix(dfs: list, genes: set, class1: str, class2: str,
                   source_label: str) -> pd.DataFrame:
    """Merge cohort dataframes, filter to panel genes, return sample×gene+meta."""
    data = pd.concat(dfs, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]

    sub = data[data["Gene"].isin(genes)].copy()
    sub = sub.drop(columns=["ILMN_ID"], errors="ignore")

    expr_cols = sub.columns.drop("Gene")
    sub["__mean__"] = sub[expr_cols].mean(axis=1)
    sub = sub.sort_values("__mean__", ascending=False)
    sub = sub.drop_duplicates(subset="Gene", keep="first")
    sub = sub.drop(columns="__mean__")
    sub = sub.set_index("Gene")

    X = sub.T.apply(pd.to_numeric, errors="coerce")

    mask = (
        X.index.str.contains(rf"\b{class1}\b", regex=True) |
        X.index.str.contains(rf"\b{class2}\b", regex=True)
    )
    X = X[mask]
    y = X.index.str.contains(rf"\b{class1}\b", regex=True).astype(int)

    out = X.copy()
    out["Class"] = y
    out["Source"] = source_label
    return out


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

SOURCE_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                  "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]

def _scatter_group(ax, pc_df, source, cls, label, color, marker):
    s = pc_df[(pc_df["Source"] == source) & (pc_df["Class"] == cls)]
    ax.scatter(s["PC1"], s["PC2"], label=label,
               color=color, marker=marker, alpha=0.65, s=30, linewidths=0)


def plot_train_only(pc_df, task_name, class1, class2, outdir):
    fig, ax = plt.subplots(figsize=(6, 6))
    subset = pc_df[pc_df["Source"] == "Train"]
    colors = ["#4C72B0", "#DD8452"]
    for (cls, label), color in zip([(0, class2), (1, class1)], colors):
        s = subset[subset["Class"] == cls]
        ax.scatter(s["PC1"], s["PC2"], label=f"Train — {label}",
                   color=color, alpha=0.65, s=30, linewidths=0)
    ax.set_title(f"{task_name} — Training cohorts")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    path = outdir / f"{task_name}_PCA_train.tiff"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path.name}")


def plot_single_cohort(pc_df, task_name, cohort_name, class1, class2, outdir):
    fig, ax = plt.subplots(figsize=(6, 6))
    subset = pc_df[pc_df["Source"] == cohort_name]
    colors = ["#4C72B0", "#DD8452"]
    for (cls, label), color in zip([(0, class2), (1, class1)], colors):
        s = subset[subset["Class"] == cls]
        ax.scatter(s["PC1"], s["PC2"], label=f"{cohort_name} — {label}",
                   color=color, alpha=0.65, s=30, linewidths=0)
    ax.set_title(f"{task_name} — {cohort_name}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    path = outdir / f"{task_name}_PCA_{cohort_name}.tiff"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path.name}")


def plot_combined(pc_df, task_name, class1, class2, outdir):
    sources = sorted(pc_df["Source"].unique())
    markers = {class1: "o", class2: "^"}
    fig, ax = plt.subplots(figsize=(7, 6))

    for si, source in enumerate(sources):
        color = SOURCE_PALETTE[si % len(SOURCE_PALETTE)]
        for cls, label in [(0, class2), (1, class1)]:
            s = pc_df[(pc_df["Source"] == source) & (pc_df["Class"] == cls)]
            ax.scatter(s["PC1"], s["PC2"],
                       label=f"{source} — {label}",
                       color=color,
                       marker=markers[label],
                       alpha=0.65, s=30, linewidths=0)

    ax.set_title(f"{task_name} — All cohorts")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", frameon=False, fontsize=8,
              ncol=2 if len(sources) > 2 else 1)
    fig.tight_layout()
    path = outdir / f"{task_name}_PCA_combined.tiff"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path.name}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_task(task_name: str, cfg: dict, data_root: Path,
             panel_root: Path, outdir: Path):

    class1, class2 = cfg["class1"], cfg["class2"]
    print(f"\n=== {task_name} ({class1} vs {class2}) ===")

    panel_path = panel_root / cfg["panel_file"]
    panel = pd.read_csv(panel_path)
    panel["Gene"] = panel["Gene"].apply(normalize_gene)
    panel_genes = set(panel["Gene"])

    train_dfs = [load_cohort(find_cohort_file(data_root, c))
                 for c in cfg["train_cohorts"]]
    train_df = extract_matrix(train_dfs, panel_genes, class1, class2, "Train")

    test_dfs_by_cohort = {}
    for cohort in cfg["test_cohorts"]:
        path = find_cohort_file(data_root, cohort)
        df = load_cohort(path)
        test_dfs_by_cohort[cohort] = extract_matrix(
            [df], panel_genes, class1, class2, cohort
        )

    combined = pd.concat([train_df] + list(test_dfs_by_cohort.values()))

    feat_cols = [c for c in train_df.columns if c not in ("Class", "Source")]
    X_train = train_df[feat_cols].fillna(0.0).values
    X_all   = combined[feat_cols].fillna(0.0).values

    pca = PCA(n_components=2)
    pca.fit(X_train)
    pcs = pca.transform(X_all)

    var1, var2 = pca.explained_variance_ratio_ * 100
    print(f"  PC1: {var1:.1f}%  PC2: {var2:.1f}%")

    pc_df = combined[["Class", "Source"]].copy().reset_index(drop=True)
    pc_df["PC1"] = pcs[:, 0]
    pc_df["PC2"] = pcs[:, 1]

    outdir.mkdir(parents=True, exist_ok=True)
    plot_train_only(pc_df, task_name, class1, class2, outdir)
    for cohort in cfg["test_cohorts"]:
        plot_single_cohort(pc_df, task_name, cohort, class1, class2, outdir)
    plot_combined(pc_df, task_name, class1, class2, outdir)


def main():

    data_root = Path(
        r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Test+Train Datasets"
    )

    panel_root = Path(
        r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Supplementary\Gene Panels"
    )

    outdir = Path(
        r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Supplementary\PCA Plots"
    )

    for task_name, cfg in TASKS.items():
        run_task(task_name, cfg, data_root, panel_root, outdir)

    print(f"\nDone. All PCA plots written to: {outdir}")


if __name__ == "__main__":
    main()