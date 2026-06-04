"""
genetic_algorithm.py
====================
Algorithme Génétique (AG) pour entraîner le NeuralAgent sur SnakeEnv.

Opérateurs implémentés :
  - Sélection     : Tournoi (tournament selection)
  - Croisement    : Uniforme (uniform crossover)
  - Mutation      : Gaussienne adaptative (std diminue avec les générations)
  - Élitisme      : Les N meilleurs individus passent directement à la génération suivante

Fonction de fitness :
  fitness = score × 100 + steps_survie × 0.5
  (équilibre nourriture + survie, évaluée sur plusieurs épisodes pour la robustesse)
"""

import numpy as np
import time
from snake_env   import SnakeEnv
from neural_agent import NeuralAgent


# ──────────────────────────────────────────────────────────────────────────────
def compute_fitness(agent, n_eval=5, seed_start=0):
    """
    Évalue la fitness d'un agent sur n_eval épisodes.

    fitness = moyenne(score × 100 + steps × 0.5)

    Plusieurs épisodes pour réduire la variance liée à la grille dynamique.
    """
    env      = SnakeEnv()
    fitnesses = []

    for ep in range(n_eval):
        obs, _ = env.reset(seed=seed_start + ep)
        done   = False
        steps  = 0

        while not done:
            action = agent.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done  = terminated or truncated
            steps += 1

        score   = info.get("score", 0)
        fitness = score * 100.0 + steps * 0.5
        fitnesses.append(fitness)

    env.close()
    return float(np.mean(fitnesses))



# ──────────────────────────────────────────────────────────────────────────────
def tournament_selection(population, fitnesses, tournament_size=3):
    """
    Sélection par tournoi : tire tournament_size individus au hasard,
    retourne le meilleur.
    """
    idx       = np.random.choice(len(population), size=tournament_size, replace=False)
    best_idx  = idx[np.argmax([fitnesses[i] for i in idx])]
    return population[best_idx]


def uniform_crossover(parent1, parent2):
    """
    Croisement uniforme : chaque gène est hérité aléatoirement de l'un des deux parents.
    """
    mask   = np.random.rand(len(parent1.genome)) < 0.5
    child_genome = np.where(mask, parent1.genome, parent2.genome)
    return NeuralAgent(genome=child_genome)


def mutate(agent, mutation_rate=0.05, mutation_std=0.3):
    """
    Mutation gaussienne : chaque gène est muté avec probabilité mutation_rate,
    en ajoutant du bruit N(0, mutation_std).
    """
    genome     = agent.genome.copy()
    mask       = np.random.rand(len(genome)) < mutation_rate
    genome[mask] += np.random.randn(mask.sum()).astype(np.float32) * mutation_std
    return NeuralAgent(genome=genome)


# ──────────────────────────────────────────────────────────────────────────────
class GeneticAlgorithm:
    """
    Algorithme Génétique complet.

    Paramètres
    ----------
    pop_size        : taille de la population
    n_generations   : nombre de générations
    elite_size      : nombre d'élites conservés directement
    tournament_size : taille des tournois pour la sélection
    crossover_prob  : probabilité d'effectuer un croisement (sinon clone)
    mutation_rate   : probabilité de mutation par gène
    mutation_std_ini: écart-type initial de la mutation
    mutation_std_end: écart-type final (décroît linéairement)
    n_eval          : épisodes par évaluation de fitness
    seed            : graine de reproductibilité
    verbose         : afficher la progression
    """

    def __init__(
        self,
        pop_size        = 50,
        n_generations   = 30,
        elite_size      = 3,
        tournament_size = 4,
        crossover_prob  = 0.7,
        mutation_rate   = 0.08,
        mutation_std_ini= 0.5,
        mutation_std_end= 0.05,
        n_eval          = 5,
        seed            = 42,
        verbose         = True,
    ):
        self.pop_size         = pop_size
        self.n_generations    = n_generations
        self.elite_size       = elite_size
        self.tournament_size  = tournament_size
        self.crossover_prob   = crossover_prob
        self.mutation_rate    = mutation_rate
        self.mutation_std_ini = mutation_std_ini
        self.mutation_std_end = mutation_std_end
        self.n_eval           = n_eval
        self.seed             = seed
        self.verbose          = verbose

        # Historique pour les courbes d'apprentissage
        self.history = {
            "best_fitness"  : [],
            "mean_fitness"  : [],
            "std_fitness"   : [],
            "best_score"    : [],
            "mean_score"    : [],
        }

        self.best_agent    = None
        self.best_fitness  = -np.inf

    # ──────────────────────────────────────────
    def _mutation_std(self, generation):
        """Décroissance linéaire de l'écart-type de mutation."""
        t   = generation / max(1, self.n_generations - 1)
        return self.mutation_std_ini + t * (self.mutation_std_end - self.mutation_std_ini)

    # ──────────────────────────────────────────
    def run(self):
        """Lance l'évolution et retourne le meilleur agent."""
        np.random.seed(self.seed)
        start_time = time.time()

        # ── 1. Population initiale ──
        population = [NeuralAgent() for _ in range(self.pop_size)]

        for gen in range(self.n_generations):
            mut_std = self._mutation_std(gen)

            # ── 2. Évaluation ──
            fitnesses = [
                compute_fitness(agent, n_eval=self.n_eval, seed_start=gen * 1000)
                for agent in population
            ]

            fitnesses_arr = np.array(fitnesses)
            sorted_idx    = np.argsort(fitnesses_arr)[::-1]  # du meilleur au pire

            # Mise à jour du meilleur global
            if fitnesses_arr[sorted_idx[0]] > self.best_fitness:
                self.best_fitness = fitnesses_arr[sorted_idx[0]]
                self.best_agent   = population[sorted_idx[0]].copy()

            # ── 3. Métriques de génération ──
            # Calculer score moyen séparément
            scores_this_gen = self._eval_scores(population, gen)

            self.history["best_fitness"].append(float(fitnesses_arr[sorted_idx[0]]))
            self.history["mean_fitness"].append(float(fitnesses_arr.mean()))
            self.history["std_fitness"].append(float(fitnesses_arr.std()))
            self.history["best_score"].append(float(max(scores_this_gen)))
            self.history["mean_score"].append(float(np.mean(scores_this_gen)))

            if self.verbose:
                elapsed = time.time() - start_time
                print(
                    f"Gen {gen+1:3d}/{self.n_generations} | "
                    f"Best fit: {fitnesses_arr[sorted_idx[0]]:7.1f} | "
                    f"Mean fit: {fitnesses_arr.mean():7.1f} | "
                    f"Best score: {max(scores_this_gen):3.0f} | "
                    f"Mut σ: {mut_std:.3f} | "
                    f"⏱ {elapsed:.0f}s"
                )

            # ── 4. Élitisme ──
            new_population = [population[i].copy() for i in sorted_idx[:self.elite_size]]

            # ── 5. Sélection + Croisement + Mutation ──
            while len(new_population) < self.pop_size:
                p1 = tournament_selection(population, fitnesses, self.tournament_size)

                if np.random.rand() < self.crossover_prob:
                    p2    = tournament_selection(population, fitnesses, self.tournament_size)
                    child = uniform_crossover(p1, p2)
                else:
                    child = p1.copy()

                child = mutate(child,
                               mutation_rate=self.mutation_rate,
                               mutation_std=mut_std)
                new_population.append(child)

            population = new_population

        if self.verbose:
            print(f"\n✅ Évolution terminée en {time.time()-start_time:.1f}s")
            print(f"   Meilleur fitness global : {self.best_fitness:.1f}")

        return self.best_agent

    # ──────────────────────────────────────────
    def _eval_scores(self, population, gen, n=5):
        """Évalue le score brut (nourriture) pour les métriques."""
        env    = SnakeEnv()
        scores = []
        for agent in population:
            ep_scores = []
            for ep in range(n):
                obs, _ = env.reset(seed=gen * 1000 + ep + 9999)
                done   = False
                while not done:
                    obs, _, terminated, truncated, info = env.step(agent.act(obs))
                    done = terminated or truncated
                ep_scores.append(info.get("score", 0))
            scores.append(np.mean(ep_scores))
        env.close()
        return scores


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Lancement de l'AG (version rapide : 10 générations, pop=20) ...")
    ag = GeneticAlgorithm(
        pop_size     = 20,
        n_generations= 10,
        elite_size   = 2,
        n_eval       = 3,
        seed         = 42,
        verbose      = True,
    )
    best = ag.run()
    print(f"\nMeilleur agent : genome_size={best.genome_size}")
