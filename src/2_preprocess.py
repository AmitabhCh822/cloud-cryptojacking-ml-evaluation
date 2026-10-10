"""Run both preprocessing pipelines and report the dataset statistics in Section 4.

DS2OS:   drop the seven leakage-prone columns, label encode, stratified 70/30
         split (seed 42), SMOTE on the training part only, standardize.
NSL-KDD: official KDDTrain+/KDDTest+ partition, label encoding fitted on both
         partitions, no resampling, standardize with training statistics.

Usage:
    python src/2_preprocess.py

Author: Amitabh Chakravorty
"""

import json

from common import (
    METRICS, ds2os_features, ds2os_split, ds2os_target, ensure_dirs,
    load_ds2os, load_nsl, nsl_encode,
)


def main():
    ensure_dirs()
    stats = {}

    df = load_ds2os()
    y = ds2os_target(df)
    X = ds2os_features(df)
    X_tr, X_te, y_tr, y_te, n_syn = ds2os_split(X, y)
    stats["ds2os"] = {
        "rows": int(len(df)),
        "raw_features": int(df.shape[1] - 1),
        "features_after_removal": int(X.shape[1]),
        "feature_names": list(X.columns),
        "attack_types": sorted(t for t in df["normality"].unique() if t != "normal"),
        "normal": int((y == 0).sum()),
        "attack": int((y == 1).sum()),
        "train_after_smote": int(len(y_tr)),
        "smote_synthetic_samples": int(n_syn),
        "test_normal": int((y_te == 0).sum()),
        "test_attack": int((y_te == 1).sum()),
    }

    train, test = load_nsl()
    Xn_tr, Xn_te, yn_tr, yn_te = nsl_encode(train, test)
    stats["nsl_kdd"] = {
        "train_rows": int(len(yn_tr)),
        "test_rows": int(len(yn_te)),
        "features": int(Xn_tr.shape[1]),
        "train_normal": int((yn_tr == 0).sum()),
        "train_attack": int((yn_tr == 1).sum()),
        "test_normal": int((yn_te == 0).sum()),
        "test_attack": int((yn_te == 1).sum()),
        "attack_types_only_in_test": sorted(
            set(test["label"]) - set(train["label"])
        ),
    }

    out = METRICS / "dataset_statistics.json"
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
