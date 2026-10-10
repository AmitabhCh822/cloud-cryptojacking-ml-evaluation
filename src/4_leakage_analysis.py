"""Leakage screen and ablation on DS2OS (Section 5.3, Table 4 and Fig. 1).

Steps
  1. Single-feature lookup accuracy L(f) and base-rate-adjusted lift L*(f)
     for all twelve raw features (Eqs. 1 and 2, Table 4).
  2. Ablation: remove features one at a time in descending L* order, rerun
     the full pipeline with XGBoost and record accuracy and attack-class F1
     (Fig. 1).
  3. Anchor run of the clean five-feature pipeline (should match Table 2).
  4. Boundary check: remove destinationServiceAddress from the clean
     pipeline and report the change for all six models.

Outputs
    results/metrics/table4_leakage_scores.csv
    results/metrics/fig1_ablation_values.csv
    results/metrics/destination_service_address_check.csv
    results/figures/leakage_ablation.png

Usage:
    python src/4_leakage_analysis.py

Author: Amitabh Chakravorty
"""

import pandas as pd

from common import (
    DS2OS_REMOVED, DS2OS_TARGET, FIGURES, METRICS, MODEL_ORDER, SEED,
    build_models, ds2os_features, ds2os_split, ds2os_target, ensure_dirs,
    evaluate, load_ds2os, log_environment,
)
from plots import plot_ablation


def leakage_scores(df, y):
    """L(f): accuracy of a lookup table that predicts the majority class of each value."""
    p_maj = max(y.mean(), 1 - y.mean())
    rows = []
    for f in [c for c in df.columns if c != DS2OS_TARGET]:
        values = df[f].astype(object).where(df[f].notna(), "nan").astype(str)
        counts = pd.crosstab(values, y)
        L = counts.max(axis=1).sum() / len(y)
        rows.append({
            "feature": f,
            "L": L,
            "L_star": (L - p_maj) / (1 - p_maj),
            "n_unique": int(values.nunique()),
            "disposition": "removed" if f in DS2OS_REMOVED else "retained",
        })
    table = pd.DataFrame(rows).sort_values(["L_star", "L"], ascending=False)
    return table.reset_index(drop=True), p_maj


def run_xgb(df, y, removed, seed=SEED):
    X_tr, X_te, y_tr, y_te, _ = ds2os_split(ds2os_features(df, removed), y, seed=seed)
    model = build_models(seed, names=["XGBoost"])["XGBoost"]
    model.fit(X_tr, y_tr)
    return evaluate(y_te, model.predict(X_te))


def main():
    ensure_dirs()
    log_environment("leakage_analysis")
    df = load_ds2os()
    y = ds2os_target(df)

    # 1. Leakage screen
    table, p_maj = leakage_scores(df, y)
    table.to_csv(METRICS / "table4_leakage_scores.csv", index=False)
    print(f"Majority-class base rate p_maj = {p_maj:.4f}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 2. Ablation in descending L* order
    order = table["feature"].tolist()
    ablation = []
    for k in range(len(order)):
        m = run_xgb(df, y, order[:k])
        ablation.append({
            "n_removed": k,
            "last_removed": order[k - 1] if k else "",
            "n_features_left": len(order) - k,
            "accuracy": m["accuracy"],
            "f1_attack": m["f1_attack"],
        })
        print(f"removed {k:2d} ({ablation[-1]['last_removed'] or 'none':26s}) "
              f"acc={m['accuracy']:.5f} F1a={m['f1_attack']:.3f}")
    abl = pd.DataFrame(ablation)
    abl.to_csv(METRICS / "fig1_ablation_values.csv", index=False)
    plot_ablation(abl["n_removed"], abl["accuracy"], abl["f1_attack"],
                  FIGURES / "leakage_ablation.png")

    # 3. Anchor: clean five-feature pipeline (Table 2, XGBoost row)
    clean = run_xgb(df, y, DS2OS_REMOVED)
    print(f"\nClean pipeline XGBoost: acc={clean['accuracy']:.4f} F1a={clean['f1_attack']:.3f}")

    # 4. destinationServiceAddress boundary check, all six models
    check = []
    for variant, removed in (("clean", DS2OS_REMOVED),
                             ("clean_minus_destinationServiceAddress",
                              DS2OS_REMOVED + ["destinationServiceAddress"])):
        X_tr, X_te, y_tr, y_te, _ = ds2os_split(ds2os_features(df, removed), y)
        for name, model in build_models().items():
            model.fit(X_tr, y_tr)
            m = evaluate(y_te, model.predict(X_te))
            check.append({"variant": variant, "model": name,
                          "accuracy": m["accuracy"], "f1_attack": m["f1_attack"]})
    chk = pd.DataFrame(check).pivot(index="model", columns="variant")
    chk.columns = [f"{a}_{b}" for a, b in chk.columns]
    chk["delta_accuracy_points"] = 100 * (
        chk["accuracy_clean_minus_destinationServiceAddress"] - chk["accuracy_clean"])
    chk["delta_f1_attack"] = (
        chk["f1_attack_clean_minus_destinationServiceAddress"] - chk["f1_attack_clean"])
    chk = chk.reindex(MODEL_ORDER)
    chk.to_csv(METRICS / "destination_service_address_check.csv")
    print("\nRemoving destinationServiceAddress from the clean pipeline:")
    print(chk[["delta_accuracy_points", "delta_f1_attack"]].round(4).to_string())


if __name__ == "__main__":
    main()
