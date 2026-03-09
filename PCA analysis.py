import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

# CONFIG

TASK_NAME = "ATB_vs_LTB"  # ATB_vs_LTB, ATB_vs_OD, LTB_vs_CON

TASKS = {
    "ATB_vs_LTB": ("ATB", "LTB"),
    "ATB_vs_OD":  ("ATB", "OD"),
    "LTB_vs_CON": ("LTB", "CON"),
}

CLASS1, CLASS2 = TASKS[TASK_NAME]

PANEL_CSV = r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Supplementary\Gene Panels\Native 20 genes.csv"

TRAIN_FILES = [
    r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Test+Train Datasets\Train Datasets\Train Datasets Normalized\GSE19491_zscored_final.csv",
    r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Test+Train Datasets\Train Datasets\Train Datasets Normalized\GSE37250_zscored_final.csv",
    r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Test+Train Datasets\Train Datasets\Train Datasets Normalized\GSE73408_zscored_final.csv",
]

TEST_FILES = [
    r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Test+Train Datasets\Test Datasets\Test Datasets Normalized\GSE101705_zscored_final.csv",
    r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Test+Train Datasets\Test Datasets\Test Datasets Normalized\GSE107994_zscored_final.csv",
]

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

def extract_matrix(data, genes, source_label):

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
        X.index.str.contains(fr"\b{CLASS1}\b", regex=True) |
        X.index.str.contains(fr"\b{CLASS2}\b", regex=True)
    )

    X = X[mask]
    y = X.index.str.contains(fr"\b{CLASS1}\b", regex=True).astype(int)

    df_out = X.copy()
    df_out["Class"] = y
    df_out["Source"] = source_label

    return df_out

panel = pd.read_csv(PANEL_CSV)
panel["Gene"] = panel["Gene"].apply(normalize_gene)
panel_genes = set(panel["Gene"])

train_data = load_and_merge(TRAIN_FILES)
train_df = extract_matrix(train_data, panel_genes, "Train")

external_dfs = []
for test_file in TEST_FILES:
    cohort_name = Path(test_file).stem
    test_data = load_and_merge([test_file])
    df = extract_matrix(test_data, panel_genes, cohort_name)
    external_dfs.append(df)

combined = pd.concat([train_df] + external_dfs, axis=0)

# PCA 

X_train = train_df.drop(columns=["Class","Source"]).fillna(0.0).values
X_all = combined.drop(columns=["Class","Source"]).fillna(0.0).values

pca = PCA(n_components=2)
pca.fit(X_train)          # fit only on training data

pcs = pca.transform(X_all)   # project all samples

print("Explained variance:", pca.explained_variance_ratio_)

y_class = combined["Class"].values
y_source = combined["Source"].values

pc_df = pd.DataFrame({
    "PC1": pcs[:,0],
    "PC2": pcs[:,1],
    "Class": y_class,
    "Source": y_source
})

# PLOT 1 — TRAIN 

plt.figure(figsize=(6,6))
subset = pc_df[pc_df["Source"] == "Train"]

for cls,label in [(0,CLASS2),(1,CLASS1)]:
    s = subset[subset["Class"] == cls]
    plt.scatter(s["PC1"], s["PC2"], label=f"Train {label}", alpha=0.6)

plt.title(f"{TASK_NAME} — Train Only")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig(f"{TASK_NAME}_PCA_train.tiff", dpi=600, bbox_inches="tight")
plt.show()

# PLOT 2 & 3 EXTERNAL COHORTS 

for test_file in sorted(TEST_FILES):

    cohort_name = Path(test_file).stem
    plt.figure(figsize=(6,6))

    subset = pc_df[pc_df["Source"] == cohort_name]

    for cls,label in [(0,CLASS2),(1,CLASS1)]:
        s = subset[subset["Class"] == cls]
        plt.scatter(
            s["PC1"], s["PC2"],
            label=f"{cohort_name} {label}",
            alpha=0.6
        )

    plt.title(f"{TASK_NAME} — {cohort_name} Only")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{TASK_NAME}_PCA_{cohort_name}.tiff", dpi=600, bbox_inches="tight")
    plt.show()

#  PLOT 4 — COMBINED (4-COLOR VIEW) 

plt.figure(figsize=(6,6))

for source in sorted(pc_df["Source"].unique()):
    for cls,label in [(0,CLASS2),(1,CLASS1)]:

        s = pc_df[(pc_df["Source"]==source) & (pc_df["Class"]==cls)]

        plt.scatter(
            s["PC1"], s["PC2"],
            label=f"{source}-{label}",
            alpha=0.6
        )

plt.title(f"{TASK_NAME} — Combined (Source + Class)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig(f"{TASK_NAME}_PCA_combined.tiff", dpi=600, bbox_inches="tight")
plt.show()