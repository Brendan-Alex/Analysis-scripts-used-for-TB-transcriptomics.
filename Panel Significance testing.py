import pandas as pd
import numpy as np
from pathlib import Path

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

PANELS = {
    "Native20": r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Supplementary\Gene Panels\Native 20 genes.csv",
    "Zak16": r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Supplementary\Gene Panels\Zak16.csv",
    "Berry86": r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Supplementary\Gene Panels\Berry86_clean.csv",
    "Sweeney3": r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Supplementary\Gene Panels\Sweeney3.csv"
}

USE_RANDOM_PANEL = True
RANDOM_PANEL_SIZE = 20

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
np.random.seed(SEED)

#  MODEL 

def get_model():
    return MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=SEED)

#  HELPERS 

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

#  DATA PREP 

train_data = load_and_merge(TRAIN_FILES)
all_train_genes = available_genes(TRAIN_FILES)
all_test_genes  = available_genes(TEST_FILES)
gene_universe = all_train_genes & all_test_genes

panel_sets = {}

for panel_name, panel_path in PANELS.items():
    panel_df = pd.read_csv(panel_path)
    panel_df["Gene"] = panel_df["Gene"].apply(normalize_gene)
    panel_sets[panel_name] = set(panel_df["Gene"])

if USE_RANDOM_PANEL:
    rng = np.random.default_rng(SEED)
    random_genes = set(rng.choice(list(gene_universe), size=RANDOM_PANEL_SIZE, replace=False))
    panel_sets[f"Random{RANDOM_PANEL_SIZE}_seed{SEED}"] = random_genes

#  TEST + SIGNIFICANCE 

for test_file in TEST_FILES:

    cohort_name = Path(test_file).stem
    print("\n===============")
    print(f"TASK: {TASK_NAME} | COHORT: {cohort_name}")
    print("===============")

    test_data = load_and_merge([test_file])

    panel_preds = {}
    panel_aucs = {}

    for panel_name, panel_genes in panel_sets.items():

        print(f"\n--- Panel: {panel_name} ---")

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

        model = get_model()
        model.fit(X_train, y_train)

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

        panel_preds[panel_name] = p_test
        panel_aucs[panel_name] = roc_auc_score(y_test, p_test)

        print(f"AUC: {panel_aucs[panel_name]:.3f}")

    print("\nAUCs:", panel_aucs)

    # ===== Significance vs Native20 =====

    if "Native20" not in panel_preds:
        print("Native20 missing — skipping significance.")
        continue

    baseline = panel_preds["Native20"]
    results = []

    for pname, preds in panel_preds.items():
        if pname == "Native20":
            continue

        delta, lo, hi, p = bootstrap_auc_test(
        y_test,
        baseline,
        preds,
        n_boot=1000,
        seed=SEED
        )

        results.append({
            "Comparison": f"{pname} vs Native20",
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
            OUTDIR / f"{TASK_NAME}_PANEL_SIGNIFICANCE_{cohort_name}.csv",
            index=False
        )

        print("\nSignificance vs Native20:")
        print(results_df)

print("\n PANEL SIGNIFICANCE TESTING COMPLETE")
