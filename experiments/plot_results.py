"""Publication figures (300 DPI, svg/pdf/jpg) from the benchmark CSVs.

fig1  compression trajectory  -- the iWSP loop prunes the metadata leak (FPR drops)
fig2  adversarial robustness  -- ASR vs epsilon, unconstrained and physics-bounded
fig3  operating point         -- false-alarm rate vs detection F1, sized by params
fig4  security scorecard      -- FPR, backdoor ASR, adversarial ASR (the real wins)

Run after run_benchmark.py:  PYTHONPATH=. python experiments/plot_results.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "figures")

FIG_BG, AX_BG = "#0f0f1a", "#1a1a2e"
AMBER, CORAL, CYAN, EMERALD, PURPLE = "#ffd166", "#ff6b6b", "#4cc9f0", "#06d6a0", "#b983ff"
FG, GRID = "#e8e8f0", "#33334d"

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 11,
    "figure.facecolor": FIG_BG, "axes.facecolor": AX_BG,
    "savefig.facecolor": FIG_BG, "text.color": FG, "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG, "axes.edgecolor": GRID,
    "grid.color": GRID, "axes.titlecolor": FG,
})


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("svg", "pdf", "jpg"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig1_trajectory():
    df = pd.read_csv(os.path.join(RESULTS, "wsp_trajectory.csv"))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.grid(True, alpha=0.3)
    ax.plot(df["round"], df["hidden_f1"], "o-", color=AMBER, lw=2.4, label="held-out macro F1")
    ax.set_xlabel("iWSP revision round"); ax.set_ylabel("held-out macro F1", color=AMBER)
    ax.tick_params(axis="y", colors=AMBER)
    ax2 = ax.twinx()
    ax2.plot(df["round"], df["fpr"], "s--", color=CORAL, lw=2.0, label="false-positive rate")
    ax2.plot(df["round"], df["params"] / df["params"].max() * df["fpr"].max(),
             ":", color=CYAN, lw=1.5, label="params (scaled)")
    ax2.set_ylabel("false-positive rate", color=CORAL); ax2.tick_params(axis="y", colors=CORAL)
    ax.axvspan(-0.4, 1.5, color=CORAL, alpha=0.08)
    ax.annotate("metadata-leak detectors\nfire on benign -> pruned",
                (0.5, df["hidden_f1"].min()), color=CORAL, fontsize=8.5, ha="center")
    for _, r in df.iterrows():
        ax.annotate(r["name"].split("_", 1)[-1], (r["round"], r["hidden_f1"]),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=7.5, color=FG, alpha=0.8)
    ax.set_title("iWSP revision: pruning the leak collapses false positives (0.61 -> 0.01)",
                 fontsize=10.8)
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="center right", facecolor=AX_BG,
              edgecolor=GRID, fontsize=8.5)
    _save(fig, "fig1_compression_trajectory")


def fig2_robustness():
    df = pd.read_csv(os.path.join(RESULTS, "asr_curves.csv"))
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.grid(True, alpha=0.3)
    ax.plot(df.epsilon, df.mlp_pgd, "o-", color=CORAL, lw=1.8, label="MLP (white-box PGD)")
    ax.plot(df.epsilon, df.cnn1d_pgd, "^-", color=PURPLE, lw=1.6, label="1D-CNN (white-box PGD)")
    ax.plot(df.epsilon, df.resnet_pgd, "v-", color=CYAN, lw=1.6, label="ResNet-tab (white-box PGD)")
    ax.plot(df.epsilon, df.iwsp_transfer_pgd, "s--", color=AMBER, lw=2.4, label="iWSP (transfer PGD)")
    ax.plot(df.epsilon, df.iwsp_physics, "D-", color=EMERALD, lw=2.6, label="iWSP (physics-bounded FDI)")
    ax.set_xlabel(r"perturbation budget $\epsilon$ (per-feature, $\times$ train std)")
    ax.set_ylabel("attack success rate (evasion)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Evasion robustness on real VeReMi: invariants resist what gradients cannot",
                 fontsize=10.5)
    ax.legend(loc="center right", facecolor=AX_BG, edgecolor=GRID, fontsize=8.5)
    _save(fig, "fig2_adversarial_robustness")


def fig3_operating_point():
    df = pd.read_csv(os.path.join(RESULTS, "model_comparison.csv"))
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.grid(True, alpha=0.3)
    for _, r in df.iterrows():
        passed = bool(r["audit_passed"])
        color = AMBER if passed else CORAL
        marker = "*" if r["name"] == "iWSP-compact" else "o"
        size = 460 if r["name"] == "iWSP-compact" else 150
        ax.scatter(r["false_positive_rate"], r["hidden_detect_f1"], s=size, marker=marker,
                   color=color, edgecolor=FG, linewidth=0.8, zorder=3)
        ax.annotate(f"{r['name']}\n({int(r['n_params'])}p, "
                    f"{'audit OK' if passed else 'opaque'})",
                    (r["false_positive_rate"], r["hidden_detect_f1"]),
                    textcoords="offset points", xytext=(9, -3), fontsize=8, color=FG)
    ax.axvspan(0.10, 0.30, color=CORAL, alpha=0.08)
    ax.annotate("unusable false-alarm\nregion for a real IDS", (0.20, 0.90),
                color=CORAL, fontsize=8.5, ha="center")
    ax.set_xlabel("false-positive rate (benign flagged)")
    ax.set_ylabel("held-out detection F1 (attack vs. benign)")
    ax.set_title("Operating point: iWSP detects at 0.7% false alarms; gradient models at 14-28%",
                 fontsize=10.2)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="*", color="none", markerfacecolor=AMBER,
               markeredgecolor=FG, markersize=15, label="22-param iWSP (audit passes)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CORAL,
               markeredgecolor=FG, markersize=10, label="opaque baselines (audit fails)")],
        loc="lower right", facecolor=AX_BG, edgecolor=GRID, fontsize=8.5)
    _save(fig, "fig3_param_f1_pareto")


def fig4_security():
    comp = pd.read_csv(os.path.join(RESULTS, "model_comparison.csv"))
    bd = pd.read_csv(os.path.join(RESULTS, "backdoor.csv"))
    asr = pd.read_csv(os.path.join(RESULTS, "asr_curves.csv"))
    bd_map = {r["model"].split(" ")[0]: r["backdoor_asr"] for _, r in bd.iterrows()}
    # adversarial ASR at eps=1.0
    row1 = asr.iloc[(asr.epsilon - 1.0).abs().idxmin()]
    adv = {"iWSP-compact": row1["iwsp_physics"], "mlp": row1["mlp_pgd"],
           "cnn1d": row1["cnn1d_pgd"], "resnet_tab": row1["resnet_pgd"]}
    names = ["iWSP-compact", "mlp", "cnn1d", "resnet_tab"]
    labels = ["iWSP", "MLP", "1D-CNN", "ResNet"]
    fpr = [comp.loc[comp.name == n, "false_positive_rate"].iloc[0] for n in names]
    bdr = [bd_map.get(n, 0.0) for n in names]
    adr = [adv[n] for n in names]

    x = np.arange(len(names)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.grid(True, axis="y", alpha=0.3)
    ax.bar(x - w, fpr, w, color=CYAN, label="false-positive rate")
    ax.bar(x, bdr, w, color=PURPLE, label="backdoor ASR (poisoned train)")
    ax.bar(x + w, adr, w, color=CORAL, label=r"adversarial ASR ($\epsilon{=}1$)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("rate (lower is better)")
    ax.set_title("Security scorecard on real VeReMi: the 22-param iWSP dominates every axis",
                 fontsize=10.5)
    ax.legend(loc="upper left", facecolor=AX_BG, edgecolor=GRID, fontsize=8.5)
    _save(fig, "fig4_security_scorecard")


def main():
    fig1_trajectory(); fig2_robustness(); fig3_operating_point(); fig4_security()
    print(f"figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
