import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, brier_score_loss,
    confusion_matrix, roc_curve
)

#  CONFIG 

TASK_NAME = "LTB_vs_CON"  # ATB_vs_LTB, ATB_vs_OD, LTB_vs_CON

TASKS = {
    "ATB_vs_LTB": ("ATB", "LTB"),
    "ATB_vs_OD":  ("ATB", "OD"),
    "LTB_vs_CON": ("LTB", "CON"),
}

CLASS1, CLASS2 = TASKS[TASK_NAME]

PANEL_CSV = r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Supplementary\Gene Panels\Native 20 genes.csv"

USE_RANDOM_PANEL = False
RANDOM_PANEL_SIZE = 20

TRAIN_FILES = [
    r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Test+Train Datasets\Train Datasets\Train Datasets Normalized\GSE19439_zscored_final.csv",
    r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Test+Train Datasets\Train Datasets\Train Datasets Normalized\GSE19444_zscored_final.csv",
    r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Test+Train Datasets\Train Datasets\Train Datasets Normalized\GSE28623_zscored_final.csv"
]

OUTDIR = Path(
    r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Supplementary\Final Benchmarking Results\Model Comparison\Internal\LTB v CON"
)
OUTDIR.mkdir(exist_ok=True, parents=True)

SEED = 42
THRESH = 0.5
N_BOOT = 1000
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

#  MODELS 

MODELS = {
    "logreg": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(n_estimators=500, random_state=SEED),
    "mlp": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=SEED),
}

#  LOADERS 

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

#  METRICS 

def compute_metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p > THRESH).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv  = tp / (tp + fp) if tp + fp else np.nan
    npv  = tn / (tn + fn) if tn + fn else np.nan
    auc  = roc_auc_score(y, p)
    brier = brier_score_loss(y, p)
    bal_acc = (sens + spec) / 2
    return auc, brier, sens, spec, ppv, npv, bal_acc

#  LOCO SPLITS 

def loco_splits(files):
    for i in range(len(files)):
        yield (
            [files[j] for j in range(len(files)) if j != i],
            [files[i]]
        )

#  PANEL 

all_genes = available_genes(TRAIN_FILES)

if USE_RANDOM_PANEL:
    rng = np.random.default_rng(SEED)
    panel_genes = set(rng.choice(list(all_genes), RANDOM_PANEL_SIZE, replace=False))
    panel_name = f"Random{RANDOM_PANEL_SIZE}_seed{SEED}"
else:
    panel = pd.read_csv(PANEL_CSV)
    panel["Gene"] = panel["Gene"].apply(normalize_gene)
    panel_genes = set(panel["Gene"])
    panel_name = Path(PANEL_CSV).stem

usable = panel_genes & all_genes

#  RUN INTERNAL LOCO 

model_curves = {}

for model_name, model in MODELS.items():
    print(f"\n=== INTERNAL LOCO: {model_name.upper()} | {TASK_NAME} ===")

    all_preds, all_labels = [], []

    for fold, (train_files, test_files) in enumerate(loco_splits(TRAIN_FILES), start=1):
        print(f"LOCO fold {fold}: test = {Path(test_files[0]).stem}")

        train_data = load_and_merge(train_files)
        test_data  = load_and_merge(test_files)

        X_train, y_train = extract_X_y(train_data, usable, CLASS1, CLASS2)
        X_test,  y_test  = extract_X_y(test_data, usable, CLASS1, CLASS2)

        X_test = X_test.reindex(columns=X_train.columns)
        X_train = X_train.fillna(0.0)
        X_test  = X_test.fillna(0.0)

        X_train = X_train.to_numpy()
        X_test  = X_test.to_numpy()

        model.fit(X_train, y_train)
        p_test = model.predict_proba(X_test)[:, 1]

        all_preds.append(p_test)
        all_labels.append(y_test)

    y_all = np.concatenate(all_labels)
    p_all = np.concatenate(all_preds)

    model_curves[model_name] = (y_all, p_all)

    auc, brier, sens, spec, ppv, npv, bal_acc = compute_metrics(y_all, p_all)

    run = f"{TASK_NAME}_{model_name}_{panel_name}_INTERNAL_LOCO"

    pd.DataFrame({
        "Sample": range(len(p_all)),
        "TrueLabel": y_all,
        "PredProb": p_all,
        "Task": TASK_NAME,
        "Model": model_name,
        "Panel": panel_name
    }).to_csv(OUTDIR / f"{run}_test_predictions.csv", index=False)

    print(f"AUC={auc:.3f} | BalAcc={bal_acc:.3f}")

#  COMBINED ROC 

plt.figure(figsize=(6,6))

for model_name, (y, p) in model_curves.items():
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)

    plt.plot(
        fpr, tpr,
        color=MODEL_COLORS.get(model_name, "black"),
        linewidth=2,
        label=f"{model_name.upper()} (AUC = {auc:.3f})"
    )

# Diagonal reference
plt.plot([0,1], [0,1], linestyle="--", color="gray", linewidth=1.5)

plt.xlim(0,1)
plt.ylim(0,1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"{TASK_NAME} — Internal LOCO (Models Compared)")

plt.legend(loc="lower right", frameon=False)

plt.tight_layout()
plt.savefig(
    OUTDIR / f"{TASK_NAME}_INTERNAL_models_ROC.tiff",
    dpi=600,
    bbox_inches="tight"
)
plt.close()

print("\n INTERNAL LOCO COMPLETE — strict pairwise classification enforced")