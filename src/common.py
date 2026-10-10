"""Shared pipeline code for the cloud cryptojacking evaluation.

Every script in src/ imports from this module so that preprocessing,
model configuration and metrics are identical across all experiments.

Author: Amitabh Chakravorty
"""

import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTEN
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
FIGURES = RESULTS / "figures"

SEED = 42
TEST_SIZE = 0.3

# ---------------------------------------------------------------------------
# DS2OS
# ---------------------------------------------------------------------------
DS2OS_FILES = ("DS2OS.csv", "mainSimulationAccessTraces.csv")
DS2OS_TARGET = "normality"

# Columns removed by the leakage-aware pipeline (Table 4 of the paper).
DS2OS_REMOVED = [
    "timestamp",
    "value",
    "sourceID",
    "sourceAddress",
    "accessedNodeAddress",
    "accessedNodeType",
    "operation",
]

# ---------------------------------------------------------------------------
# NSL-KDD
# ---------------------------------------------------------------------------
NSL_TRAIN_FILE = "KDDTrain+.txt"
NSL_TEST_FILE = "KDDTest+.txt"
NSL_CATEGORICAL = ["protocol_type", "service", "flag"]
NSL_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

MODEL_ORDER = [
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "Gradient Boosting",
    "Decision Tree",
    "KNN",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dirs():
    for d in (METRICS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)


def find_file(names, data_dir=DATA_RAW):
    """Return the first file under data_dir whose name matches one of names."""
    if isinstance(names, str):
        names = (names,)
    data_dir = Path(data_dir)
    for name in names:
        hits = sorted(data_dir.rglob(name))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"Could not find {' or '.join(names)} under {data_dir}. "
        "Run src/1_download_data.py first or place the file there manually."
    )


def build_models(seed=SEED, names=None):
    """The six configurations listed in Table 1 of the paper."""
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=20, min_samples_split=5,
            random_state=seed, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=10, learning_rate=0.1,
            random_state=seed, n_jobs=-1, eval_metric="logloss",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100, max_depth=10,
            random_state=seed, n_jobs=-1, verbose=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=seed,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15, min_samples_split=5, random_state=seed,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }
    if names:
        models = {k: v for k, v in models.items() if k in names}
    return models


def evaluate(y_true, y_pred):
    """Support-weighted metrics (Table 2) and attack-class metrics (Table 3)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_attack": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_attack": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_attack": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def log_environment(tag):
    """Record library versions and hardware for the run that produced the results."""
    import imblearn
    import lightgbm
    import matplotlib
    import sklearn
    import xgboost

    ensure_dirs()
    info = {
        "script": tag,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
        "imbalanced-learn": imblearn.__version__,
        "matplotlib": matplotlib.__version__,
    }
    path = RESULTS / f"environment_{tag}.json"
    path.write_text(json.dumps(info, indent=2))
    return info


# ---------------------------------------------------------------------------
# DS2OS pipeline
# ---------------------------------------------------------------------------
def load_ds2os(data_dir=DATA_RAW):
    return pd.read_csv(find_file(DS2OS_FILES, data_dir), low_memory=False)


def ds2os_target(df):
    """Binary target: normal = 0, any anomalous class = 1."""
    return (df[DS2OS_TARGET] != "normal").astype(int).to_numpy()


def ds2os_features(df, removed=DS2OS_REMOVED):
    """Drop the given feature columns and label encode the categorical ones."""
    X = df.drop(columns=[DS2OS_TARGET] + list(removed)).copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            values = X[col].astype(object).where(X[col].notna(), "nan").astype(str)
            X[col] = LabelEncoder().fit_transform(values)
    if X.isnull().to_numpy().any():
        X = X.fillna(X.mean())
    return X


def ds2os_split(X, y, seed=SEED, sampler="smote"):
    """Stratified 70/30 split, oversampling on the training part only, then scaling."""
    imbalance = y.sum() / max(len(y) - y.sum(), 1)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    n_synthetic = 0
    if imbalance < 0.3 or imbalance > 3.0:
        k = min(5, int(y_tr.sum()) - 1)
        before = len(y_tr)
        if sampler == "smote":
            X_tr, y_tr = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X_tr, y_tr)
        elif sampler == "smoten":
            X_tr, y_tr = SMOTEN(random_state=seed, k_neighbors=k).fit_resample(X_tr, y_tr)
        elif sampler != "none":
            raise ValueError(f"Unknown sampler {sampler}")
        n_synthetic = len(y_tr) - before
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, X_te, np.asarray(y_tr), np.asarray(y_te), n_synthetic


# ---------------------------------------------------------------------------
# NSL-KDD pipeline
# ---------------------------------------------------------------------------
def load_nsl(data_dir=DATA_RAW):
    train = pd.read_csv(find_file(NSL_TRAIN_FILE, data_dir), names=NSL_COLUMNS)
    test = pd.read_csv(find_file(NSL_TEST_FILE, data_dir), names=NSL_COLUMNS)
    return train.drop(columns=["difficulty"]), test.drop(columns=["difficulty"])


def nsl_encode(train, test):
    """Binary labels and label encoding fitted on the union of both partitions."""
    X_tr = train.drop(columns=["label"]).copy()
    X_te = test.drop(columns=["label"]).copy()
    for col in NSL_CATEGORICAL:
        le = LabelEncoder().fit(pd.concat([X_tr[col], X_te[col]]).astype(str))
        X_tr[col] = le.transform(X_tr[col].astype(str))
        X_te[col] = le.transform(X_te[col].astype(str))
    y_tr = (train["label"] != "normal").astype(int).to_numpy()
    y_te = (test["label"] != "normal").astype(int).to_numpy()
    return X_tr.to_numpy(dtype=float), X_te.to_numpy(dtype=float), y_tr, y_te


def scale(X_tr, X_te):
    scaler = StandardScaler()
    return scaler.fit_transform(X_tr), scaler.transform(X_te)
