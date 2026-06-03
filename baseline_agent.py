"""
baseline_agent.py
=================
Baseline de comparaison pour le projet Snake M2442.

Deux baselines :
  1. RandomAgent    : choisit une action aléatoire à chaque pas.
  2. HeuristicAgent : va vers la nourriture en évitant les dangers immédiats.
"""

import numpy as np
from snake_env import SnakeEnv


# ──────────────────────────────────────────────────────────────────────────────
class RandomAgent:
    """Agent qui choisit une action uniformément aléatoire."""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        return self.action_space.sample()


# ──────────────────────────────────────────────────────────────────────────────
class HeuristicAgent:
    """
    Agent heuristique simple basé sur l'observation de 12 features.

    Logique :
      1. Éliminer les actions qui causent une collision immédiate.
      2. Parmi les actions sûres, préférer celle qui rapproche de la nourriture.
      3. Si toutes les actions sont dangereuses, choisir aléatoirement.
    """

    # Mapping direction courante → (action_relative_gauche, action_relative_droite)
    # (même logique que l'observation)
    DIR_TO_FOOD = {
        0: (6, 7),   # Haut  → nourriture gauche=idx6, droite=idx7
        1: (7, 6),   # Bas   → inverser
        2: (5, 4),   # Gauche→ nourriture bas=idx5,   haut=idx4
        3: (4, 5),   # Droite→ nourriture haut=idx4,  bas=idx5
    }

    def act(self, obs):
        """
        obs[0:4]  = dangers (Haut, Bas, Gauche, Droite)
        obs[4:8]  = nourriture (haut, bas, gauche, droite)
        obs[8:12] = direction one-hot
        """
        dangers   = obs[0:4]
        food_dir  = obs[4:8]  # [up, down, left, right]

        # Direction actuelle
        current_dir = int(np.argmax(obs[8:12]))

        # Actions sûres
        safe_actions = [a for a in range(4) if dangers[a] == 0]

        if not safe_actions:
            return int(np.random.randint(4))

        # Scorer chaque action sûre selon la nourriture
        # food_dir: 0=haut, 1=bas, 2=gauche, 3=droite
        # action:   0=haut, 1=bas, 2=gauche, 3=droite  (même mapping)
        best_action = safe_actions[0]
        best_score  = -1

        for a in safe_actions:
            s = food_dir[a]
            if s > best_score:
                best_score  = s
                best_action = a

        return best_action


# ──────────────────────────────────────────────────────────────────────────────
def evaluate_agent(agent, n_episodes=100, seed_start=0, render=False):
    """
    Évalue un agent sur n_episodes épisodes.

    Retourne un dictionnaire de métriques :
      - scores        : liste des scores par épisode
      - survivals     : liste des durées de survie (steps)
      - mean_score    : moyenne des scores
      - mean_survival : moyenne des steps
      - max_score     : meilleur score obtenu
    """
    env    = SnakeEnv(render_mode="ansi" if render else None)
    scores    = []
    survivals = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_start + ep)
        done = False
        total_steps = 0

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_steps += 1

            if render and ep == 0:
                env.render()

        scores.append(info.get("score", 0))
        survivals.append(total_steps)

    env.close()
    return {
        "scores"       : scores,
        "survivals"    : survivals,
        "mean_score"   : float(np.mean(scores)),
        "std_score"    : float(np.std(scores)),
        "mean_survival": float(np.mean(survivals)),
        "std_survival" : float(np.std(survivals)),
        "max_score"    : int(np.max(scores)),
    }


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("=" * 50)
    print("ÉVALUATION DES BASELINES (100 épisodes)")
    print("=" * 50)

    # ── Baseline 1 : Aléatoire ──
    env_tmp = SnakeEnv()
    random_agent = RandomAgent(env_tmp.action_space)
    env_tmp.close()

    r_metrics = evaluate_agent(random_agent, n_episodes=100, seed_start=0)
    print("\n📊 Agent Aléatoire :")
    print(f"  Score moyen    : {r_metrics['mean_score']:.2f} ± {r_metrics['std_score']:.2f}")
    print(f"  Survie moyenne : {r_metrics['mean_survival']:.1f} steps")
    print(f"  Meilleur score : {r_metrics['max_score']}")

    # ── Baseline 2 : Heuristique ──
    heuristic_agent = HeuristicAgent()
    h_metrics = evaluate_agent(heuristic_agent, n_episodes=100, seed_start=0)
    print("\n📊 Agent Heuristique :")
    print(f"  Score moyen    : {h_metrics['mean_score']:.2f} ± {h_metrics['std_score']:.2f}")
    print(f"  Survie moyenne : {h_metrics['mean_survival']:.1f} steps")
    print(f"  Meilleur score : {h_metrics['max_score']}")

    print("\n✅ Baselines évaluées !")
