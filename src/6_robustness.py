"""Stability and oversampling checks (Section 3.3 and Section 4.2).

1. Seed stability: rerun every model with ten random seeds. On DS2OS the
   seed changes the split, SMOTE and the model; on NSL-KDD the holdout is
   fixed, so the seed changes only the model.
2. SMOTE-N: rerun the DS2OS clean pipeline with SMOTE-N, which treats the
   label-encoded features as categorical, and compare with SMOTE.

Outputs
    results/metrics/seed_stability.csv
    results/metrics/smoten_vs_smote.csv

Usage:
    python src/6_robustness.py [--seeds 0 1 2 3 4 5 6 7 8 9]

Author: Amitabh Chakravorty
"""

import argparse

import pandas as pd

from common import (
    METRICS, MODEL_ORDER, build_models, ds2os_features, ds2os_split,
    ds2os_target, ensure_dirs, evaluate, load_ds2os, load_nsl, log_environment,
    nsl_encode, scale,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args()

    ensure_dirs()
    log_environment("robustness")
    df = load_ds2os()
    y = ds2os_target(df)
    X = ds2os_features(df)
    train, test = load_nsl()
    Xn_tr, Xn_te, yn_tr, yn_te = nsl_encode(train, test)
    Xn_tr, Xn_te = scale(Xn_tr, Xn_te)

    # 1. Seed stability
    runs = []
    for seed in args.seeds:
        X_tr, X_te, y_tr, y_te, _ = ds2os_split(X, y, seed=seed)
        for name, model in build_models(seed).items():
            model.fit(X_tr, y_tr)
            runs.append({"dataset": "DS2OS", "model": name, "seed": seed,
                         "accuracy": evaluate(y_te, model.predict(X_te))["accuracy"]})
        for name, model in build_models(seed).items():
            model.fit(Xn_tr, yn_tr)
            runs.append({"dataset": "NSL-KDD", "model": name, "seed": seed,
                         "accuracy": evaluate(yn_te, model.predict(Xn_te))["accuracy"]})
        print(f"seed {seed} done")
    runs = pd.DataFrame(runs)
    summary = (runs.groupby(["dataset", "model"])["accuracy"]
               .agg(mean="mean", std="std").reset_index())
    summary["std_points"] = 100 * summary["std"]
    summary.round(5).to_csv(METRICS / "seed_stability.csv", index=False)
    print(summary.round(4).to_string(index=False))
    for dataset in ("DS2OS", "NSL-KDD"):
        worst = summary[summary.dataset == dataset]["std_points"].max()
        print(f"Largest accuracy standard deviation on {dataset}: {worst:.3f} points")

    # 2. SMOTE-N against SMOTE on the DS2OS clean pipeline
    rows = []
    for sampler in ("smote", "smoten"):
        X_tr, X_te, y_tr, y_te, _ = ds2os_split(X, y, sampler=sampler)
        for name, model in build_models().items():
            model.fit(X_tr, y_tr)
            m = evaluate(y_te, model.predict(X_te))
            rows.append({"sampler": sampler, "model": name,
                         "accuracy": m["accuracy"], "f1_attack": m["f1_attack"]})
    comp = pd.DataFrame(rows).pivot(index="model", columns="sampler")
    comp.columns = [f"{a}_{b}" for a, b in comp.columns]
    comp["delta_accuracy_points"] = 100 * (comp["accuracy_smoten"] - comp["accuracy_smote"])
    comp["delta_f1_attack"] = comp["f1_attack_smoten"] - comp["f1_attack_smote"]
    comp = comp.reindex(MODEL_ORDER)
    comp.round(5).to_csv(METRICS / "smoten_vs_smote.csv")
    print("\nSMOTE-N minus SMOTE:")
    print(comp[["delta_accuracy_points", "delta_f1_attack"]].round(4).to_string())


if __name__ == "__main__":
    main()
