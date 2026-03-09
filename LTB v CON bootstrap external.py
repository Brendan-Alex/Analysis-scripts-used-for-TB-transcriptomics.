import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, roc_auc_score

# CONFIG

TASKS = {
    "LTB_vs_CON": {"positive": "LTB", "negative": "CON"},
    "ATB_vs_OD": {"positive": "ATB", "negative": "OD"},
    "ATB_vs_LTB": {"positive": "ATB", "negative": "LTB"}
}

TASK = "LTB_vs_CON"
POS_LABEL = TASKS[TASK]["positive"]
NEG_LABEL = TASKS[TASK]["negative"]

RANDOM_PANEL_SIZE = 20
N_RANDOM = 50
SEED = 42

PANEL_PATHS = {
    "Native20": r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Supplementary\Gene Panels\Native 20 genes.csv",
    "Berry86": r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Supplementary\Gene Panels\Berry86_clean.csv",
    "Zak16": r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Supplementary\Gene Panels\Zak16.csv",
    "Sweeney3": r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Supplementary\Gene Panels\Sweeney3.csv"
}

TRAIN_FILES = [
    r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Test+Train Datasets\Train Datasets\Train Datasets Normalized\GSE19439_zscored_final.csv",
    r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Test+Train Datasets\Train Datasets\Train Datasets Normalized\GSE19444_zscored_final.csv",
    r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Test+Train Datasets\Train Datasets\Train Datasets Normalized\GSE28623_zscored_final.csv",
]

TEST_FILES = [
    r"C:\Users\brend\Desktop\Project Data\Combined final dataset\Test+Train Datasets\Test Datasets\Test Datasets Normalized\GSE107994_zscored_final.csv"
]

rng = np.random.default_rng(SEED)

def normalize_gene(g):
    return str(g).strip().upper().split(".")[0]

def load_and_merge(files):
    dfs=[]
    for f in files:
        df=pd.read_csv(f)
        df.columns=df.columns.str.strip()
        df["Gene"]=df["Gene"].apply(normalize_gene)
        dfs.append(df)
    data=pd.concat(dfs,axis=1)
    return data.loc[:,~data.columns.duplicated()]

def available_genes(files):
    gene_sets=[]
    for f in files:
        df=pd.read_csv(f)
        df["Gene"]=df["Gene"].apply(normalize_gene)
        gene_sets.append(set(df["Gene"]))
    return set.intersection(*gene_sets)

def extract_X_y(data,genes):

    sub=data[data["Gene"].isin(genes)].copy()
    sub=sub.drop(columns=["ILMN_ID"],errors="ignore")

    expr=sub.columns.drop("Gene")
    sub["__mean__"]=sub[expr].mean(axis=1)
    sub=sub.sort_values("__mean__",ascending=False)

    sub=sub.drop_duplicates("Gene").drop(columns="__mean__").set_index("Gene")

    X=sub.T
    X=X[~X.index.astype(str).str.startswith("ILMN_")]
    X=X.apply(pd.to_numeric,errors="coerce")

    labels=pd.Series(X.index.astype(str),index=X.index)
    pos=labels.str.contains(POS_LABEL)
    neg=labels.str.contains(NEG_LABEL)

    valid=pos|neg
    X=X.loc[valid]
    y=pos.loc[valid].astype(int)

    if y.nunique()<2:
        raise ValueError(f"Only one class present for task {TASK}")

    return X,y

def get_model():
    return MLPClassifier(hidden_layer_sizes=(50,),max_iter=500,random_state=SEED)

# LOAD DATA

train_data=load_and_merge(TRAIN_FILES)
test_data=load_and_merge(TEST_FILES)
common_genes=available_genes(TRAIN_FILES)&available_genes(TEST_FILES)

# RANDOM ROC CLOUD

plt.figure(figsize=(6,6))
random_aucs=[]

for _ in range(N_RANDOM):

    panel=set(rng.choice(list(common_genes),size=RANDOM_PANEL_SIZE,replace=False))

    X_train,y_train=extract_X_y(train_data,panel)
    X_test,y_test=extract_X_y(test_data,panel)

    X_test=X_test.reindex(columns=X_train.columns)

    X_train=X_train.fillna(0).to_numpy()
    X_test=X_test.fillna(0).to_numpy()

    model=get_model()
    model.fit(X_train,y_train)

    p=model.predict_proba(X_test)[:,1]

    auc=roc_auc_score(y_test,p)
    random_aucs.append(auc)

    fpr,tpr,_=roc_curve(y_test,p)

    plt.plot(fpr,tpr,color="black",alpha=0.12,linewidth=0.8)

# CURATED PANELS 

colors={
    "Native20":"#3B82F6",
    "Sweeney3":"#F0E442",
    "Zak16":"#D55E00",
    "Berry86":"#009E73"
}

curated_aucs={}

for name in sorted(PANEL_PATHS):

    panel=pd.read_csv(PANEL_PATHS[name])
    panel["Gene"]=panel["Gene"].apply(normalize_gene)

    genes=set(panel["Gene"])&common_genes
    print(name,"gene count used:",len(genes))

    X_train,y_train=extract_X_y(train_data,genes)
    X_test,y_test=extract_X_y(test_data,genes)

    X_test=X_test.reindex(columns=X_train.columns)

    X_train=X_train.fillna(0).to_numpy()
    X_test=X_test.fillna(0).to_numpy()

    model=get_model()
    model.fit(X_train,y_train)

    p=model.predict_proba(X_test)[:,1]
    auc=roc_auc_score(y_test,p)

    curated_aucs[name]=auc

    fpr,tpr,_=roc_curve(y_test,p)

    plt.plot(
        fpr,tpr,
        color=colors.get(name,"black"),
        linewidth=1.8,
        label=f"{name} (AUC={auc:.2f})",
        zorder=5
    )

# PLOT SETUP

plt.plot([0,1],[0,1],"--",color="dimgray",linewidth=1)

plt.xlim(0,1)
plt.ylim(0,1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(f"{TASK.replace('_',' ')} — External ROC: Random {RANDOM_PANEL_SIZE} 50 Bootstraps vs Curated Panels")

plt.legend(loc="lower right",frameon=False)

plt.tight_layout()

plt.savefig(
    f"{TASK}_ROC_cloud_external.tiff",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

# STATISTICS 

random_aucs=np.array(random_aucs)

print("\nCurated AUCs:")
for k in sorted(curated_aucs):
    print(f"{k}: {curated_aucs[k]:.3f}")

print("\nRandom distribution:")
print(f"Mean AUC: {random_aucs.mean():.3f}")
print(f"SD AUC: {random_aucs.std(ddof=1):.3f}")

print("\nEmpirical P(Random ≥ Panel):")
for k in sorted(curated_aucs):
    print(f"{k}: {np.mean(random_aucs>=curated_aucs[k]):.3f}")