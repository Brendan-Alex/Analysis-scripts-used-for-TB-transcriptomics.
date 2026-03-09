import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

#CONFIG

ROOT = Path(r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Supplementary\Final Benchmarking results")
OUT_DIR = Path(r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Supplementary\AUC Plots")
OUT_DIR.mkdir(exist_ok=True)

PANEL_COLORS = {
    "Native20": "#3B82F6",
    "Sweeney3": "#F0E442",
    "Zak16": "#D55E00",
    "Kaforou": "#CC79A7",
    "Berry86": "#009E73",
}

def load_all_probs(root):

    dfs = []
    scanned = 0

    for f in root.rglob("*"):

        if not f.is_file():
            continue

        if f.suffix.lower() not in [".csv", ".tsv", ".txt"]:
            continue

        scanned += 1

        for sep in [",", "\t"]:

            try:
                df = pd.read_csv(f, sep=sep)
            except Exception:
                continue

            required = {"TrueLabel", "PredProb", "Task", "Model", "Panel"}

            if not required.issubset(df.columns):
                continue

            evaluation = "External" if "External" in f.parts else "Internal"

            df = df[["TrueLabel", "PredProb", "Task", "Model", "Panel"]].copy()
            df["Evaluation"] = evaluation

            dfs.append(df)
            break

    print(f"Scanned {scanned} files, loaded {len(dfs)} probability tables")

    if not dfs:
        raise RuntimeError("No probability files found with required columns!")

    df = pd.concat(dfs, ignore_index=True)

    # Normalize model names

    df["Model"] = (
        df["Model"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Normalize panel names 

    df["Panel"] = (
        df["Panel"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "", regex=True)
    )

    df.loc[df["Panel"].str.contains("native"), "Panel"] = "Native20"
    df.loc[df["Panel"].str.contains("zak"), "Panel"] = "Zak16"
    df.loc[df["Panel"].str.contains("berry"), "Panel"] = "Berry86"
    df.loc[df["Panel"].str.contains("sweeney"), "Panel"] = "Sweeney3"
    df.loc[df["Panel"].str.contains("kaforou"), "Panel"] = "Kaforou"

    return df


def plot_models_within_task(df, task_label):

    plt.figure(figsize=(6, 6))

    models = sorted(df["Model"].dropna().unique())

    for model in models:

        msub = df[df["Model"] == model]

        if msub["TrueLabel"].nunique() < 2:
            continue

        fpr, tpr, _ = roc_curve(msub["TrueLabel"], msub["PredProb"])

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model} (AUC={auc(fpr, tpr):.3f})"
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(f"{task_label}: Models compared (Panel = Native20)")

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.legend(loc="lower right")

    plt.tight_layout()

    safe = task_label.replace(" ", "_").replace("(", "").replace(")", "")

    plt.savefig(
        OUT_DIR / f"{safe}_models_ROC.tiff",
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()


def plot_panels_within_task(df, task_label):

    plt.figure(figsize=(6, 6))

    panels = sorted(df["Panel"].dropna().unique())

    for panel in panels:

        psub = df[df["Panel"] == panel]

        if psub["TrueLabel"].nunique() < 2:
            continue

        fpr, tpr, _ = roc_curve(psub["TrueLabel"], psub["PredProb"])

        color = PANEL_COLORS.get(panel, "#000000")

        if isinstance(panel, str) and "random" in panel.lower():
            color = "#000000"

        plt.plot(
            fpr,
            tpr,
            color=color,
            linewidth=2,
            label=f"{panel} (AUC={auc(fpr, tpr):.3f})"
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(f"{task_label}: Panels compared (Model = MLP)")

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.legend(loc="lower right", fontsize=8)

    plt.tight_layout()

    safe = task_label.replace(" ", "_").replace("(", "").replace(")", "")

    plt.savefig(
        OUT_DIR / f"{safe}_panels_ROC.tiff",
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()


# ANALYSIS

df = load_all_probs(ROOT)

print("Tasks:", sorted(df.Task.unique()))
print("Evaluations:", sorted(df.Evaluation.unique()))
print("Models:", sorted(df.Model.unique()))
print("Panels:", sorted(df.Panel.unique()))

for evaluation in sorted(df["Evaluation"].unique()):

    for task in sorted(df["Task"].unique()):

        sub = df[(df["Evaluation"] == evaluation) & (df["Task"] == task)]

        if sub.empty:
            continue

        model_sub = sub[sub["Panel"] == "Native20"]

        if not model_sub.empty:
            label = f"{task} ({evaluation})"
            plot_models_within_task(model_sub, label)

        panel_sub = sub[sub["Model"] == "MLP"]

        if not panel_sub.empty:
            label = f"{task} ({evaluation})"
            plot_panels_within_task(panel_sub, label)

print("Done. ROC plots saved to:", OUT_DIR)