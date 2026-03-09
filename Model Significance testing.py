import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix
from statsmodels.stats.multitest import multipletests

#  CONFIG 

TASK_NAME = "LTB_vs_CON"

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
    r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Supplementary\dump"
)
OUTDIR.mkdir(exist_ok=True, parents=True)

SEED = 42
THRESH = 0.5
np.random.seed(SEED)

#  MODELS 

MODELS = {
    "logreg": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(n_estimators=500, random_state=SEED),
    "mlp": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=SEED),
}

def bootstrap_auc_test(y_true, pred_base, pred_other, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        p1_b = pred_base[idx]
        p2_b = pred_other[idx]

        if len(np.unique(y_b)) < 2:
            continue

        auc1 = roc_auc_score(y_b, p1_b)
        auc2 = roc_auc_score(y_b, p2_b)
        diffs.append(auc2 - auc1)

    diffs = np.array(diffs)

    return (
        np.mean(diffs),
        np.percentile(diffs, 2.5),
        np.percentile(diffs, 97.5),
        2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    )

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

    if len(np.unique(y)) != 2:
        raise ValueError("ERROR: Not exactly two classes present.")

    return X, y

#  RUN 

panel = pd.read_csv(PANEL_CSV)
panel["Gene"] = panel["Gene"].apply(normalize_gene)
panel_genes = set(panel["Gene"])
panel_name = Path(PANEL_CSV).stem

train_data = load_and_merge(TRAIN_FILES)
all_train_genes = available_genes(TRAIN_FILES)

usable_train_genes = panel_genes & all_train_genes

X_train_df, y_train = extract_X_y(
    train_data,
    usable_train_genes,
    CLASS1,
    CLASS2
)

X_train_df = X_train_df.fillna(0.0)
feature_order = X_train_df.columns
X_train = X_train_df.to_numpy()

print("\nTraining complete.")

#  TEST + SIGNIFICANCE 

for test_file in TEST_FILES:

    cohort_name = Path(test_file).stem
    print(f"\n=== Testing on {cohort_name} ===")

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

    cohort_preds = {}

    # Train + predict
    for mname, model in MODELS.items():
        model.fit(X_train, y_train)
        cohort_preds[mname] = model.predict_proba(X_test)[:, 1]

    # AUCs
    aucs = {m: roc_auc_score(y_test, p) for m, p in cohort_preds.items()}
    print("AUCs:", aucs)

    # Significance vs MLP
    baseline = cohort_preds["mlp"]
    results = []

    for m in cohort_preds:
        if m == "mlp":
            continue

        delta, lo, hi, p = bootstrap_auc_test(
            y_test,
            baseline,
            cohort_preds[m],
            n_boot=1000,
            seed=SEED
        )

        results.append({
            "Comparison": f"{m} vs mlp",
            "Delta_AUC": delta,
            "CI_lower": lo,
            "CI_upper": hi,
            "p_value": p
        })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df["p_adj_FDR"] = multipletests(
            results_df["p_value"],
            method="fdr_bh"
        )[1]

        results_df.to_csv(
            OUTDIR / f"{TASK_NAME}_MODEL_SIGNIFICANCE_{cohort_name}.csv",
            index=False
        )

        print("\nSignificance vs MLP:")
        print(results_df)

print("\n MODEL SIGNIFICANCE TESTING COMPLETE")
