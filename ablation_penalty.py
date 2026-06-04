"""
ablation_penalty.py
===================
Étude d'ablation : Ajout d'une pénalité pour le temps sans nourriture.
AUCUNE MODIFICATION DES FICHIERS ORIGINAUX.
"""

import numpy as np
import time
from snake_env import SnakeEnv
from neural_agent import NeuralAgent
from baseline_agent import HeuristicAgent, evaluate_agent
from genetic_algorithm import tournament_selection, uniform_crossover, mutate


# ═══════════════════════════════════════════════════════════════
# 1. NOUVELLE FONCTION DE FITNESS (Avec pénalité de faim)
# ═══════════════════════════════════════════════════════════════

def get_fitness_hybrid_penalty(agent, n_eval=10, seed_start=0):
    """
    Fitness hybride : 
    - Récompense la nourriture (score × 200)
    - Récompense la survie (steps × 0.2)
    - PÉNALISE le temps passé sans manger (-0.05 par step sans manger)
    """
    env = SnakeEnv()
    fitnesses = []
    
    for i in range(n_eval):
        obs, _ = env.reset(seed=seed_start + i)
        done = False
        steps = 0
        last_score = 0
        steps_without_food = 0
        
        while not done:
            action = agent.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
            current_score = info.get("score", 0)
            
            # Si le serpent a mangé, on reset le compteur de faim
            if current_score > last_score:
                steps_without_food = 0
                last_score = current_score
            else:
                steps_without_food += 1
        
        # Calcul de la fitness finale
        # Pénalité qui augmente progressivement

        base_score = (last_score * 200.0) + (steps * 0.2)
        starvation_penalty = steps_without_food * 0.01  # Pénalité progressive
        
        fitness = base_score - starvation_penalty
        fitnesses.append(fitness)
        
    env.close()
    return float(np.mean(fitnesses))


def get_fitness_balanced(agent, n_eval=3, seed_start=0):
    """Fitness équilibrée classique (pour comparaison)."""
    env = SnakeEnv()
    fitnesses = []
    for i in range(n_eval):
        obs, _ = env.reset(seed=seed_start + i)
        done = False
        steps = 0
        while not done:
            action = agent.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            score = info.get("score", 0)
        fitness = score * 100.0 + steps * 0.5
        fitnesses.append(fitness)
    env.close()
    return float(np.mean(fitnesses))


# ═══════════════════════════════════════════════════════════════
# 2. BOUCLE D'ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════

def train_custom_ag(fitness_func, pop_size, n_generations=30, verbose=False):
    """Entraîne un AG avec une fonction de fitness personnalisée."""
    population = [NeuralAgent() for _ in range(pop_size)]
    best_agent = population[0]
    best_fitness = -99999
    
    if verbose: print(f"  ➡️ Démarrage (Pop={pop_size})...")
    
    for gen in range(n_generations):
        # Évaluation
        fitnesses = [fitness_func(ind, seed_start=gen*100) for ind in population]
        
        # Mise à jour du meilleur
        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > best_fitness:
            best_fitness = fitnesses[current_best_idx]
            best_agent = population[current_best_idx].copy()
            
        # Sélection, Croisement, Mutation
        new_population = []
        new_population.append(population[current_best_idx].copy()) # Élitisme
        
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
    print("=" * 80)
    print(" 🧪 ÉTUDE D'ABLATION : Impact de la Pénalité de Faim (Starvation Penalty)")
    print("=" * 80)
    
    # A. Évaluer l'Heuristique (Référence)
    print("\n⏳ Évaluation de l'agent Heuristique (Référence)...")
    heuristic_agent = HeuristicAgent()
    h_metrics = evaluate_agent(heuristic_agent, n_episodes=100, seed_start=5000)
    print(f"   🎯 Score Heuristique : {h_metrics['mean_score']:.2f} | Survie : {h_metrics['mean_survival']:.0f} steps\n")
    
    # B. Configurations à tester
    # On compare la fitness classique vs la fitness avec pénalité de faim
    fitness_configs = [
        ("balanced_classique", get_fitness_balanced),
        ("hybrid_penalty", get_fitness_hybrid_penalty),
    ]
    
    pop_size = 120  # Taille de population fixe pour isoler l'effet de la fitness
    n_generations = 300 # 30 générations pour aller vite, tu peux mettre 50
    
    results = []
    
    # C. Lancer les expériences
    for name, fitness_func in fitness_configs:
        start_time = time.time()
        
        print(f"🚀 Entraînement avec fitness : {name}")
        best_agent = train_custom_ag(
            fitness_func=fitness_func, 
            pop_size=pop_size, 
            n_generations=n_generations, 
            verbose=True
        )
        
        # Évaluation finale
        ag_metrics = evaluate_agent(best_agent, n_episodes=100, seed_start=5000)
        
        results.append({
            "Fitness": name,
            "Score AG": ag_metrics['mean_score'],
            "Survie AG": ag_metrics['mean_survival'],
            "Temps": round(time.time() - start_time, 1)
        })

    # D. Affichage du tableau final
    print("\n" + "=" * 80)
    print(" 📊 RÉSULTATS FINAUX (Comparaison avec l'Heuristique)")
    print("=" * 80)
    print(f"{'Fitness':<20} | {'Score AG':<10} | {'Survie AG':<12} | {'Score H':<10} | {'Survie H':<12} | {'Temps':<6}")
    print("-" * 90)
    
    for r in results:
        print(f"{r['Fitness']:<20} | {r['Score AG']:<10.2f} | {r['Survie AG']:<12.0f} | {h_metrics['mean_score']:<10.2f} | {h_metrics['mean_survival']:<12.0f} | {r['Temps']:<6}s")
        
    print("\n✅ Étude terminée !")

if __name__ == "__main__":
    main()