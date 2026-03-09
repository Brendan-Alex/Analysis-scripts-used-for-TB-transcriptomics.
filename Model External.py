import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix, roc_curve

import matplotlib.pyplot as plt

# CONFIG

TASK_NAME = "LTB_vs_CON"  # ATB_vs_LTB, ATB_vs_OD, LTB_vs_CON

TASKS = {
    "ATB_vs_LTB": ("ATB", "LTB"),
    "ATB_vs_OD":  ("ATB", "OD"),
    "LTB_vs_CON": ("LTB", "CON"),
}

CLASS1, CLASS2 = TASKS[TASK_NAME]

PANEL_CSV = r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Supplementary\Gene Panels\Native 20 genes.csv"

TRAIN_FILES = [
    r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Test+Train Datasets\Train Datasets\Train Datasets Normalized\GSE19439_zscored_final.csv",
    r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Test+Train Datasets\Train Datasets\Train Datasets Normalized\GSE19444_zscored_final.csv",
    r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Test+Train Datasets\Train Datasets\Train Datasets Normalized\GSE28623_zscored_final.csv",
]

TEST_FILES = [
    r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Test+Train Datasets\Test Datasets\Test Datasets Normalized\GSE107994_zscored_final.csv",
]

OUTDIR = Path(
    r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Supplementary\Final Benchmarking Results\Model Comparison\External\LTB v CON"
)
OUTDIR.mkdir(exist_ok=True, parents=True)

SEED = 42
THRESH = 0.5
np.random.seed(SEED)

# PLOT CONFIG

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
})

MODEL_COLORS = {
    "logreg": "#0072B2",  # Blue
    "rf": "#E60000",      # Red
    "mlp": "#F6FF00",     # Yellow
}

# MODELS 

MODELS = {
    "logreg": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(n_estimators=500, random_state=SEED),
    "mlp": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=SEED),
}

# LOADERS 

def normalize_gene(g):
    return str(g).strip().upper().split(".")[0]

def load_and_merge(files):
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        df["Gene"] = df["Gene"].apply(normalize_gene)
        dfs.append(df)
    data = pd.concat(dfs, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    return data

def available_genes(files):
    sets = []
    for f in files:
        df = pd.read_csv(f)
        df["Gene"] = df["Gene"].apply(normalize_gene)
        sets.append(set(df["Gene"]))
    return set.intersection(*sets)


def extract_X_y(data, genes, class1, class2):
    sub = data[data["Gene"].isin(genes)].copy()
    sub = sub.drop(columns=["ILMN_ID"], errors="ignore")

    expr_cols = sub.columns.drop("Gene")
    sub["__mean__"] = sub[expr_cols].mean(axis=1)
    sub = sub.sort_values("__mean__", ascending=False)
    sub = sub.drop_duplicates(subset="Gene", keep="first")
    sub = sub.drop(columns="__mean__")
    sub = sub.set_index("Gene")

    X = sub.T
    X = X.apply(pd.to_numeric, errors="coerce")

    mask = (
        X.index.str.contains(class1) |
        X.index.str.contains(class2)
    )

    X = X[mask]
    y = X.index.str.contains(class1).astype(int)

    counts = np.bincount(y)
    print(f"Class counts ({class1}=1 vs {class2}=0):", counts)

    if len(counts) != 2:
        raise ValueError("ERROR: Not exactly two classes present.")

    return X, y

def compute_metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p > THRESH).ravel()

    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv  = tp / (tp + fp) if (tp + fp) else np.nan
    npv  = tn / (tn + fn) if (tn + fn) else np.nan

    auc = roc_auc_score(y, p)
    brier = brier_score_loss(y, p)

    return auc, brier, sens, spec, ppv, npv

def plot_roc(y_true, y_prob, model_name, cohort_name, auc_value):
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6, 6))

    plt.plot(
        fpr, tpr,
        color=MODEL_COLORS.get(model_name, "black"),
        linewidth=2,
        label=f"{model_name.upper()} (AUC = {auc_value:.3f})"
    )

    # Diagonal reference
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{TASK_NAME} — External ({cohort_name})")

    plt.legend(loc="lower right", frameon=False)

    plt.tight_layout()

    outpath = OUTDIR / f"{TASK_NAME}_{model_name}_{cohort_name}_ROC.tiff"
    plt.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.close()

# ANALYSIS

all_train_genes = available_genes(TRAIN_FILES)

panel = pd.read_csv(PANEL_CSV)
panel["Gene"] = panel["Gene"].apply(normalize_gene)
panel_genes = set(panel["Gene"])
panel_name = Path(PANEL_CSV).stem

train_data = load_and_merge(TRAIN_FILES)

for model_name, model in MODELS.items():

    print("\n=============")
    print(f"MODEL: {model_name.upper()} | TASK: {TASK_NAME}")
    print("=============")

    X_train_df, y_train = extract_X_y(
        train_data,
        panel_genes & all_train_genes,
        CLASS1,
        CLASS2
    )

    X_train_df = X_train_df.fillna(0.0)
    feature_order = X_train_df.columns
    X_train = X_train_df.to_numpy()

    model.fit(X_train, y_train)

    cohort_results = []

    for test_file in TEST_FILES:

        cohort_name = Path(test_file).stem
        print(f"\n--- Testing on {cohort_name} ---")

        test_data = load_and_merge([test_file])
        usable_genes = panel_genes & available_genes([test_file]) & all_train_genes

        X_test_df, y_test = extract_X_y(
            test_data,
            usable_genes,
            CLASS1,
            CLASS2
        )

        X_test_df = X_test_df.reindex(columns=feature_order, fill_value=0.0)
        X_test = X_test_df.fillna(0.0).to_numpy()

        p_test = model.predict_proba(X_test)[:, 1]
        test_vals = compute_metrics(y_test, p_test)

        cohort_results.append(test_vals)

        auc_value = test_vals[0]
        plot_roc(y_test, p_test, model_name, cohort_name, auc_value)

        run = f"{TASK_NAME}_{model_name}_{panel_name}_{cohort_name}_EXTERNAL"

        pd.DataFrame({
            "Metric": ["AUC","Brier","Sensitivity","Specificity","PPV","NPV"],
            "Test": test_vals,
        }).to_csv(OUTDIR / f"{run}_metrics.csv", index=False)

        pd.DataFrame({
            "TrueLabel": y_test,
            "PredProb": p_test,
            "Task": TASK_NAME,
            "Model": model_name,
            "Panel": panel_name,
            "Cohort": cohort_name
        }).to_csv(OUTDIR / f"{run}_predictions.csv", index=False)

        print(f"AUC ({cohort_name}): {auc_value:.3f}")

    cohort_array = np.array(cohort_results)

    summary_df = pd.DataFrame({
        "Metric": ["AUC","Brier","Sensitivity","Specificity","PPV","NPV"],
        "Mean": cohort_array.mean(axis=0),
        "Min": cohort_array.min(axis=0),
        "Max": cohort_array.max(axis=0),
    })

    summary_df.to_csv(
        OUTDIR / f"{TASK_NAME}_{model_name}_{panel_name}_EXTERNAL_summary.csv",
        index=False
    )

    print("\nExternal Summary (Mean / Range):")
    print(summary_df)

print("\n EXTERNAL MODEL COMPARISON COMPLETE — per-cohort evaluation + summary saved")