"""Train the six models on both datasets (Tables 2 and 3 of the paper).

Outputs
    results/metrics/table2_performance.csv     accuracy, weighted F1, weighted precision, training time
    results/metrics/table3_attack_metrics.csv  attack-class precision, recall, F1, FPR, MCC
    results/metrics/confusion_counts.csv       TN, FP, FN, TP for every model
    results/figures/cost_vs_accuracy.png       training time against accuracy

Usage:
    python src/3_train_evaluate.py

Author: Amitabh Chakravorty
"""

import time

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    FIGURES, METRICS, MODEL_ORDER, build_models, ds2os_features, ds2os_split,
    ds2os_target, ensure_dirs, evaluate, load_ds2os, load_nsl, log_environment,
    nsl_encode, scale,
)


def run(dataset, X_tr, X_te, y_tr, y_te):
    rows = []
    for name, model in build_models().items():
        start = time.time()
        model.fit(X_tr, y_tr)
        train_time = time.time() - start
        metrics = evaluate(y_te, model.predict(X_te))
        rows.append({"dataset": dataset, "model": name, "train_time_s": train_time, **metrics})
        print(f"  {dataset:8s} {name:18s} acc={metrics['accuracy']:.4f} "
              f"F1w={metrics['f1_weighted']:.4f} F1a={metrics['f1_attack']:.3f} "
              f"MCC={metrics['mcc']:.3f} time={train_time:.2f}s")
    return rows


def plot_cost(df, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    styles = {"DS2OS": ("o", "#2b6cb0"), "NSL-KDD": ("s", "#2e8b57")}
    for dataset, (marker, color) in styles.items():
        part = df[df.dataset == dataset]
        ax.scatter(part.train_time_s, part.accuracy, marker=marker, s=120,
                   color=color, edgecolor="white", label=dataset, zorder=3)
        for _, r in part.iterrows():
            ax.annotate(r.model, (r.train_time_s, r.accuracy), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("Training time (seconds, log scale)")
    ax.set_ylabel("Accuracy")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    ensure_dirs()
    log_environment("train_evaluate")
    rows = []

    print("DS2OS (leakage-aware five-feature pipeline, stratified 70/30 split)")
    df = load_ds2os()
    X_tr, X_te, y_tr, y_te, _ = ds2os_split(ds2os_features(df), ds2os_target(df))
    rows += run("DS2OS", X_tr, X_te, y_tr, y_te)

    print("NSL-KDD (official KDDTrain+ / KDDTest+ holdout)")
    train, test = load_nsl()
    Xn_tr, Xn_te, yn_tr, yn_te = nsl_encode(train, test)
    Xn_tr, Xn_te = scale(Xn_tr, Xn_te)
    rows += run("NSL-KDD", Xn_tr, Xn_te, yn_tr, yn_te)

    res = pd.DataFrame(rows)
    res["model"] = pd.Categorical(res["model"], MODEL_ORDER, ordered=True)
    res = res.sort_values(["dataset", "model"])

    t2 = res[["dataset", "model", "accuracy", "f1_weighted", "precision_weighted", "train_time_s"]].copy()
    t2["accuracy_pct"] = (t2["accuracy"] * 100).round(2)
    t2.round(4).to_csv(METRICS / "table2_performance.csv", index=False)

    t3 = res[["dataset", "model", "precision_attack", "recall_attack", "f1_attack", "fpr", "mcc"]]
    t3.round(3).to_csv(METRICS / "table3_attack_metrics.csv", index=False)

    res[["dataset", "model", "tn", "fp", "fn", "tp"]].to_csv(
        METRICS / "confusion_counts.csv", index=False)

    plot_cost(res, FIGURES / "cost_vs_accuracy.png")
    print(f"\nSaved tables to {METRICS} and figure to {FIGURES}")


if __name__ == "__main__":
    main()
