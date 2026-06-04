"""
ablation_study.py
=================
Script indépendant pour l'étude d'ablation.
Il teste l'impact de la Fitness et de la Taille de Population, 
et compare les résultats avec l'Heuristique.
AUCUNE MODIFICATION DES FICHIERS ORIGINAUX N'EST NÉCESSAIRE.
"""

import numpy as np
import time
from snake_env import SnakeEnv
from neural_agent import NeuralAgent
from baseline_agent import HeuristicAgent, evaluate_agent

# Importation des opérateurs génétiques du fichier original
from genetic_algorithm import tournament_selection, uniform_crossover, mutate


# ═══════════════════════════════════════════════════════════════
# 1. FONCTION DE FITNESS PERSONNALISÉE (Sans toucher au fichier original)
# ═══════════════════════════════════════════════════════════════

def get_fitness_variant(agent, fitness_type="balanced", n_eval=3, seed_start=0):
    """Calcule la fitness selon 3 variantes différentes."""
    env = SnakeEnv()
    scores = []
    
    for i in range(n_eval):
        obs, _ = env.reset(seed=seed_start + i)
        done = False
        steps = 0
        episode_score = 0
        
        while not done:
            action = agent.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            episode_score = info.get("score", 0)
            
        # Application de la variante de fitness
        if fitness_type == "aggressive":
            fitness = episode_score * 500.0 + steps * 0.1  # Favorise la nourriture
        elif fitness_type == "survival":
            fitness = episode_score * 50.0 + steps * 1.0   # Favorise la survie
        else:  # balanced
            fitness = episode_score * 100.0 + steps * 0.5  # Équilibre
            
        scores.append(fitness)
        
    env.close()
    return float(np.mean(scores))


# ═══════════════════════════════════════════════════════════════
# 2. BOUCLE D'ENTRAÎNEMENT PERSONNALISÉE
# ═══════════════════════════════════════════════════════════════

def train_custom_ag(fitness_type, pop_size, n_generations=20, verbose=False):
    """Entraîne un AG avec une fitness et une population données."""
    # 1. Initialisation
    population = [NeuralAgent() for _ in range(pop_size)]
    best_agent = population[0]
    best_fitness = -99999
    
    if verbose: print(f"  ➡️ Démarrage (Pop={pop_size}, Fitness={fitness_type})...")
    
    # 2. Boucle des générations
    for gen in range(n_generations):
        # Évaluation
        fitnesses = [get_fitness_variant(ind, fitness_type, seed_start=gen*100) for ind in population]
        
        # Mise à jour du meilleur
        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > best_fitness:
            best_fitness = fitnesses[current_best_idx]
            best_agent = population[current_best_idx].copy()
            
        # Sélection, Croisement, Mutation (on réutilise tes fonctions originales !)
        new_population = []
        # Élitisme : on garde le meilleur
        new_population.append(population[current_best_idx].copy())
        
        # Création des enfants
        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitnesses, tournament_size=3)
            p2 = tournament_selection(population, fitnesses, tournament_size=3)
            child = uniform_crossover(p1, p2)
            child = mutate(child, mutation_rate=0.08, mutation_std=0.3)
            new_population.append(child)
            
        population = new_population
        
    return best_agent


# ═══════════════════════════════════════════════════════════════
# 3. SCRIPT PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print(" 🧪 ÉTUDE D'ABLATION : Impact de la Fitness et de la Population")
    print("=" * 70)
    
    # A. Évaluer l'Heuristique une seule fois (la référence)
    print("\n⏳ Évaluation de l'agent Heuristique (Référence)...")
    heuristic_agent = HeuristicAgent()
    h_metrics = evaluate_agent(heuristic_agent, n_episodes=100, seed_start=5000)
    print(f"   🎯 Score Heuristique : {h_metrics['mean_score']:.2f} | Survie : {h_metrics['mean_survival']:.0f} steps\n")
    
    # B. Configurations à tester
    fitness_types = ["balanced", "aggressive", "survival"]
    pop_sizes = [40, 80]  # J'ai mis 40 et 80 pour que ce soit rapide à tester
    
    results = []
    
    # C. Lancer les expériences
    for ft in fitness_types:
        for pop in pop_sizes:
            start_time = time.time()
            
            # Entraînement (20 générations pour aller vite, tu peux mettre 50 si tu as le temps)
            best_agent = train_custom_ag(fitness_type=ft, pop_size=pop, n_generations=50, verbose=True)
            
            # Évaluation finale de l'agent entraîné
            ag_metrics = evaluate_agent(best_agent, n_episodes=100, seed_start=5000)
            
            results.append({
                "Fitness": ft,
                "Population": pop,
                "Score AG": ag_metrics['mean_score'],
                "Survie AG": ag_metrics['mean_survival'],
                "Temps": round(time.time() - start_time, 1)
            })

    # D. Affichage du tableau final
    print("\n" + "=" * 70)
    print(" 📊 RÉSULTATS FINAUX (Comparaison avec l'Heuristique)")
    print("=" * 70)
    print(f"{'Fitness':<12} | {'Pop':<4} | {'Score AG':<10} | {'Survie AG':<10} | {'Score H':<10} | {'Survie H':<10} | {'Temps':<6}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['Fitness']:<12} | {r['Population']:<4} | {r['Score AG']:<10.2f} | {r['Survie AG']:<10.0f} | {h_metrics['mean_score']:<10.2f} | {h_metrics['mean_survival']:<10.0f} | {r['Temps']:<6}s")
        
    print("\n✅ Étude terminée ! Tu peux utiliser ces données pour ton rapport.")

if __name__ == "__main__":
    main()