"""Seen-attack versus novel-attack evaluation on NSL-KDD (Section 5.4, Table 5 and Fig. 2).

Acc_seen:  stratified random split of KDDTrain+, so every test attack type
           also appears in training.
Acc_novel: official KDDTest+ holdout, which adds attack types absent from
           KDDTrain+.
Both runs share one pipeline, so the gap G = Acc_seen - Acc_novel isolates
the evaluation protocol.

Fig. 2 reports XGBoost recall on every KDDTest+ attack type with at least
50 test samples.

Outputs
    results/metrics/table5_generalization_gap.csv
    results/metrics/fig2_per_class_recall.csv
    results/figures/per_class_recall.png

Usage:
    python src/5_generalization.py [--seen-test-size 0.3]

Author: Amitabh Chakravorty
"""

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from common import (
    FIGURES, METRICS, MODEL_ORDER, SEED, build_models, ensure_dirs, evaluate,
    load_nsl, log_environment, nsl_encode, scale,
)
from plots import plot_per_class

MIN_TEST_SAMPLES = 50
RARE_TRAIN_LIMIT = 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seen-test-size", type=float, default=0.3,
                        help="test fraction of the random KDDTrain+ split")
    args = parser.parse_args()

    ensure_dirs()
    log_environment("generalization")
    train, test = load_nsl()
    X_tr_raw, X_te_raw, y_tr, y_te = nsl_encode(train, test)

    # Novel-attack holdout
    X_tr, X_te = scale(X_tr_raw, X_te_raw)
    # Seen-attack random split of KDDTrain+
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        X_tr_raw, y_tr, test_size=args.seen_test_size, random_state=SEED, stratify=y_tr)
    Xs_tr, Xs_te = scale(Xs_tr, Xs_te)

    rows, xgb_pred = [], None
    for name in MODEL_ORDER:
        novel_model = build_models(names=[name])[name]
        novel_model.fit(X_tr, y_tr)
        pred = novel_model.predict(X_te)
        if name == "XGBoost":
            xgb_pred = pred
        acc_novel = evaluate(y_te, pred)["accuracy"]

        seen_model = build_models(names=[name])[name]
        seen_model.fit(Xs_tr, ys_tr)
        acc_seen = evaluate(ys_te, seen_model.predict(Xs_te))["accuracy"]

        rows.append({"model": name, "acc_seen": acc_seen, "acc_novel": acc_novel,
                     "G": acc_seen - acc_novel})
        print(f"{name:18s} seen={acc_seen:.3f} novel={acc_novel:.3f} G={acc_seen - acc_novel:.3f}")

    pd.DataFrame(rows).round(4).to_csv(METRICS / "table5_generalization_gap.csv", index=False)

    # Per-class recall of the holdout XGBoost model
    train_counts = train["label"].value_counts()
    per_class = []
    for label, n_test in test["label"].value_counts().items():
        if label == "normal" or n_test < MIN_TEST_SAMPLES:
            continue
        mask = (test["label"] == label).to_numpy()
        n_train = int(train_counts.get(label, 0))
        group = "novel" if n_train == 0 else ("rare" if n_train < RARE_TRAIN_LIMIT else "seen")
        per_class.append({"label": label, "recall": float((xgb_pred[mask] == 1).mean()),
                          "n_test": int(n_test), "n_train": n_train, "group": group})
    pc = pd.DataFrame(per_class).sort_values(["group", "recall"])
    pc.round(4).to_csv(METRICS / "fig2_per_class_recall.csv", index=False)
    print("\n" + pc.round(3).to_string(index=False))
    plot_per_class(pc, FIGURES / "per_class_recall.png")


if __name__ == "__main__":
    main()
