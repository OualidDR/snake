"""
main.py
=======
Script principal du projet M2442 – Snake évolutif.

Étapes :
  1. Entraînement de l'agent AG (paramètres complets)
  2. Évaluation finale : AG vs Aléatoire vs Heuristique
  3. Sauvegarde des résultats (JSON + génome)
  4. Affichage des courbes et tableaux

Usage :
  python main.py              # entraînement complet
  python main.py --fast       # version rapide (test)
"""

import sys
import json
import time
import numpy as np

from snake_env         import SnakeEnv
from neural_agent      import NeuralAgent
from baseline_agent    import RandomAgent, HeuristicAgent, evaluate_agent
from genetic_algorithm import GeneticAlgorithm, compute_fitness


# ──────────────────────────────────────────────────────────────────────────────
FAST_MODE = "--fast" in sys.argv

PARAMS_FULL = dict(
    pop_size        = 80,
    n_generations   = 50,
    elite_size      = 4,
    tournament_size = 5,
    crossover_prob  = 0.7,
    mutation_rate   = 0.08,
    mutation_std_ini= 0.5,
    mutation_std_end= 0.05,
    n_eval          = 5,
    seed            = 42,
    verbose         = True,
)

PARAMS_FAST = dict(
    pop_size        = 30,
    n_generations   = 20,
    elite_size      = 3,
    tournament_size = 4,
    crossover_prob  = 0.7,
    mutation_rate   = 0.08,
    mutation_std_ini= 0.5,
    mutation_std_end= 0.05,
    n_eval          = 3,
    seed            = 42,
    verbose         = True,
)

PARAMS = PARAMS_FAST if FAST_MODE else PARAMS_FULL
N_EVAL_FINAL = 200   # épisodes pour l'évaluation finale

# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" PROJET M2442 – Snake Évolutif sur Grille Dynamique")
    print("=" * 60)
    if FAST_MODE:
        print("⚡ Mode FAST activé (paramètres réduits)\n")

    # ─── 1. Entraînement AG ───────────────────────────────────────────────────
    print("\n▶ ÉTAPE 1 : Entraînement de l'Algorithme Génétique\n")
    ag         = GeneticAlgorithm(**PARAMS)
    t0         = time.time()
    best_agent = ag.run()
    train_time = time.time() - t0
    print(f"\n  Temps d'entraînement : {train_time:.1f}s")

    # ─── 2. Évaluation finale ─────────────────────────────────────────────────
    print(f"\n▶ ÉTAPE 2 : Évaluation finale ({N_EVAL_FINAL} épisodes par agent)\n")

    env_tmp = SnakeEnv()
    random_agent    = RandomAgent(env_tmp.action_space)
    env_tmp.close()
    heuristic_agent = HeuristicAgent()

    print("  Évaluation Agent Aléatoire...")
    r_metrics = evaluate_agent(random_agent,    n_episodes=N_EVAL_FINAL, seed_start=5000)

    print("  Évaluation Agent Heuristique...")
    h_metrics = evaluate_agent(heuristic_agent, n_episodes=N_EVAL_FINAL, seed_start=5000)

    print("  Évaluation Agent AG (meilleur génome)...")
    ag_metrics = evaluate_agent(best_agent,     n_episodes=N_EVAL_FINAL, seed_start=5000)

    # ─── 3. Affichage tableau comparatif ──────────────────────────────────────
    print("\n" + "=" * 60)
    print(" RÉSULTATS COMPARATIFS")
    print("=" * 60)
    header = f"{'Agent':<18} {'Score moy':>10} {'±':>6} {'Survie moy':>12} {'Max score':>10}"
    print(header)
    print("-" * 60)
    for name, m in [("Aléatoire", r_metrics),
                    ("Heuristique", h_metrics),
                    ("AG (notre agent)", ag_metrics)]:
        print(f"{name:<18} {m['mean_score']:>10.2f} {m['std_score']:>6.2f} "
              f"{m['mean_survival']:>12.1f} {m['max_score']:>10}")
    print("=" * 60)

    # Amélioration vs baselines
    if r_metrics['mean_score'] > 0:
        imp_vs_rand = (ag_metrics['mean_score'] - r_metrics['mean_score']) / r_metrics['mean_score'] * 100
    else:
        imp_vs_rand = float('inf')
    imp_vs_heur = (ag_metrics['mean_score'] - h_metrics['mean_score']) / max(h_metrics['mean_score'], 1) * 100
    print(f"\n  Amélioration AG vs Aléatoire  : {imp_vs_rand:+.1f}%")
    print(f"  Amélioration AG vs Heuristique: {imp_vs_heur:+.1f}%")

    # ─── 4. Courbe d'apprentissage (texte) ────────────────────────────────────
    print("\n▶ Courbe d'apprentissage (fitness max par génération) :\n")
    hist = ag.history["best_fitness"]
    max_val = max(hist) if hist else 1
    bar_width = 40
    for i, val in enumerate(hist):
        bar = "█" * int(val / max_val * bar_width)
        print(f"  Gen {i+1:3d} | {bar:<{bar_width}} {val:.0f}")

    # ─── 5. Sauvegarde ────────────────────────────────────────────────────────
    results = {
        "params"        : PARAMS,
        "train_time_s"  : round(train_time, 2),
        "history"       : ag.history,
        "metrics": {
            "random"    : {k: v for k, v in r_metrics.items() if k not in ("scores","survivals")},
            "heuristic" : {k: v for k, v in h_metrics.items() if k not in ("scores","survivals")},
            "ag"        : {k: v for k, v in ag_metrics.items() if k not in ("scores","survivals")},
        },
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.save("best_genome.npy", best_agent.genome)

    print("\n✅ Résultats sauvegardés : results.json + best_genome.npy")

    return best_agent, ag, results


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    best_agent, ag, results = main()
