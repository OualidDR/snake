"""
plot_results.py
===============
Génère tous les graphiques du projet à partir de results.json.

Usage :
  python plot_results.py                  # cherche results.json dans le dossier courant
  python plot_results.py --file mon_fichier.json
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ─── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor"  : "#12121e",
    "axes.facecolor"    : "#1c1c2e",
    "axes.edgecolor"    : "#3a3a5c",
    "axes.labelcolor"   : "#c8c8e0",
    "axes.titlecolor"   : "#ffffff",
    "xtick.color"       : "#888899",
    "ytick.color"       : "#888899",
    "grid.color"        : "#2a2a42",
    "grid.linestyle"    : "--",
    "grid.alpha"        : 0.6,
    "text.color"        : "#c8c8e0",
    "font.family"       : "DejaVu Sans",
    "legend.facecolor"  : "#1c1c2e",
    "legend.edgecolor"  : "#3a3a5c",
    "legend.labelcolor" : "#c8c8e0",
})

GREEN  = "#50dc78"
BLUE   = "#5ab4ff"
ORANGE = "#ffaa44"
RED    = "#ff6666"
PURPLE = "#b07aff"
DIM    = "#555570"


def load(path):
    with open(path) as f:
        return json.load(f)


# ─── Figure 1 : Courbe d'apprentissage (fitness) ─────────────────────────────
def plot_learning_curve(data, out):
    h    = data["history"]
    gens = list(range(1, len(h["best_fitness"]) + 1))
    best = h["best_fitness"]
    mean = h["mean_fitness"]
    std  = np.array(h["std_fitness"])

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#12121e")

    ax.fill_between(gens,
                    np.array(mean) - std,
                    np.array(mean) + std,
                    alpha=0.15, color=BLUE, label="±1 écart-type")
    ax.plot(gens, mean, color=BLUE,  lw=2,   label="Fitness moyenne")
    ax.plot(gens, best, color=GREEN, lw=2.5, label="Fitness max (meilleur individu)")

    # Annotation meilleur point
    idx_max = int(np.argmax(best))
    ax.annotate(f"  max = {best[idx_max]:.1f}",
                xy=(gens[idx_max], best[idx_max]),
                xytext=(gens[idx_max] + 1, best[idx_max] + 10),
                color=GREEN, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))

    ax.set_title("Courbe d'apprentissage — Fitness par génération", fontsize=14, pad=14)
    ax.set_xlabel("Génération")
    ax.set_ylabel("Fitness  (score×100 + steps×0.5)")
    ax.legend(loc="upper left")
    ax.grid(True)
    ax.set_xlim(1, len(gens))

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ {out}")


# ─── Figure 2 : Score moyen de la population par génération ──────────────────
def plot_score_evolution(data, out):
    h    = data["history"]
    gens = list(range(1, len(h["best_score"]) + 1))

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#12121e")

    ax.bar(gens, h["best_score"], color=GREEN,  alpha=0.7, label="Meilleur score (nourriture)")
    ax.plot(gens, h["mean_score"], color=ORANGE, lw=2,     label="Score moyen de la population")

    ax.set_title("Évolution du score (nourriture collectée) par génération", fontsize=14, pad=14)
    ax.set_xlabel("Génération")
    ax.set_ylabel("Score (nb. de nourritures)")
    ax.legend()
    ax.grid(True, axis="y")
    ax.set_xlim(0.5, len(gens) + 0.5)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ {out}")


# ─── Figure 3 : Comparaison des 3 agents (barres) ───────────────────────────
def plot_comparison(data, out):
    m = data["metrics"]
    agents  = ["Aléatoire", "Heuristique", "AG (évolué)"]
    colors  = [RED, ORANGE, GREEN]
    scores  = [m["random"]["mean_score"],
               m["heuristic"]["mean_score"],
               m["ag"]["mean_score"]]
    stds    = [m["random"]["std_score"],
               m["heuristic"]["std_score"],
               m["ag"]["std_score"]]
    survies = [m["random"]["mean_survival"],
               m["heuristic"]["mean_survival"],
               m["ag"]["mean_survival"]]
    maxs    = [m["random"]["max_score"],
               m["heuristic"]["max_score"],
               m["ag"]["max_score"]]

    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor("#12121e")
    gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Subplot 1 : Score moyen ──
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.bar(agents, scores, color=colors, width=0.5,
                   yerr=stds, capsize=5, error_kw={"ecolor": "white", "alpha": 0.5})
    for bar, val in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(stds) * 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=10, color="white")
    ax1.set_title("Score moyen (±σ)", fontsize=12)
    ax1.set_ylabel("Nourriture collectée / épisode")
    ax1.grid(True, axis="y")

    # ── Subplot 2 : Survie moyenne ──
    ax2 = fig.add_subplot(gs[1])
    bars2 = ax2.bar(agents, survies, color=colors, width=0.5)
    for bar, val in zip(bars2, survies):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 2,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=10, color="white")
    ax2.set_title("Survie moyenne (steps)", fontsize=12)
    ax2.set_ylabel("Nombre de steps par épisode")
    ax2.grid(True, axis="y")

    # ── Subplot 3 : Meilleur score ──
    ax3 = fig.add_subplot(gs[2])
    bars3 = ax3.bar(agents, maxs, color=colors, width=0.5)
    for bar, val in zip(bars3, maxs):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{val}", ha="center", va="bottom", fontsize=10, color="white")
    ax3.set_title("Meilleur score (max)", fontsize=12)
    ax3.set_ylabel("Score maximal obtenu")
    ax3.grid(True, axis="y")

    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle("Comparaison des agents — 200 épisodes", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ {out}")


# ─── Figure 4 : Dashboard complet (4 graphes en 1) ───────────────────────────
def plot_dashboard(data, out):
    h    = data["history"]
    m    = data["metrics"]
    gens = list(range(1, len(h["best_fitness"]) + 1))
    best = h["best_fitness"]
    mean = h["mean_fitness"]
    std  = np.array(h["std_fitness"])

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#12121e")
    fig.suptitle(
        f"Dashboard — Snake Évolutif M2442\n"
        f"pop={data['params']['pop_size']}  |  "
        f"générations={data['params']['n_generations']}  |  "
        f"seed={data['params']['seed']}  |  "
        f"durée entraînement={data['train_time_s']}s",
        fontsize=13, y=0.98
    )
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)

    # ── 1. Fitness ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(gens, np.array(mean)-std, np.array(mean)+std,
                     alpha=0.12, color=BLUE)
    ax1.plot(gens, mean, color=BLUE,  lw=1.8, label="Moyenne")
    ax1.plot(gens, best, color=GREEN, lw=2.2, label="Max")
    ax1.set_title("Fitness par génération")
    ax1.set_xlabel("Génération"); ax1.set_ylabel("Fitness")
    ax1.legend(fontsize=8); ax1.grid(True)

    # ── 2. Score (nourriture) ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(gens, h["best_score"], color=GREEN, alpha=0.6, label="Best score")
    ax2.plot(gens, h["mean_score"], color=ORANGE, lw=1.8, label="Mean score")
    ax2.set_title("Score (nourriture) par génération")
    ax2.set_xlabel("Génération"); ax2.set_ylabel("Score")
    ax2.legend(fontsize=8); ax2.grid(True, axis="y")

    # ── 3. Comparaison score moyen ──
    ax3 = fig.add_subplot(gs[1, 0])
    agents = ["Aléatoire", "Heuristique", "AG"]
    colors = [RED, ORANGE, GREEN]
    sc     = [m["random"]["mean_score"], m["heuristic"]["mean_score"], m["ag"]["mean_score"]]
    er     = [m["random"]["std_score"],  m["heuristic"]["std_score"],  m["ag"]["std_score"]]
    bars   = ax3.bar(agents, sc, color=colors, width=0.45, yerr=er,
                     capsize=5, error_kw={"ecolor":"white","alpha":0.4})
    for bar, v in zip(bars, sc):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                 f"{v:.2f}", ha="center", fontsize=9, color="white")
    ax3.set_title("Score moyen des agents (200 épisodes)")
    ax3.set_ylabel("Score moyen"); ax3.grid(True, axis="y")

    # ── 4. Comparaison survie ──
    ax4 = fig.add_subplot(gs[1, 1])
    sv   = [m["random"]["mean_survival"], m["heuristic"]["mean_survival"], m["ag"]["mean_survival"]]
    bars4 = ax4.bar(agents, sv, color=colors, width=0.45)
    for bar, v in zip(bars4, sv):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f"{v:.0f}", ha="center", fontsize=9, color="white")
    ax4.set_title("Survie moyenne des agents (200 épisodes)")
    ax4.set_ylabel("Steps de survie"); ax4.grid(True, axis="y")

    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ {out}")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="results.json")
    args = p.parse_args()

    print(f"Lecture de {args.file} ...")
    data = load(args.file)

    print("Génération des graphiques ...")
    plot_learning_curve(data, "graph_fitness.png")
    plot_score_evolution(data, "graph_scores.png")
    plot_comparison(data,      "graph_comparison.png")
    plot_dashboard(data,       "graph_dashboard.png")

    print("\n✅ 4 graphiques générés :")
    print("  • graph_fitness.png     — courbe d'apprentissage (fitness)")
    print("  • graph_scores.png      — évolution du score par génération")
    print("  • graph_comparison.png  — comparaison des 3 agents")
    print("  • graph_dashboard.png   — dashboard complet (4 en 1)")


if __name__ == "__main__":
    main()
