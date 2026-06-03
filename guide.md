# Guide du Projet — Snake Évolutif M2442

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Installation](#2-installation)
3. [Structure du projet](#3-structure-du-projet)
4. [Fichiers en détail](#4-fichiers-en-détail)
   - [snake_env.py](#41-snake_envpy)
   - [neural_agent.py](#42-neural_agentpy)
   - [baseline_agent.py](#43-baseline_agentpy)
   - [genetic_algorithm.py](#44-genetic_algorithmpy)
   - [main.py](#45-mainpy)
5. [Lancer le projet](#5-lancer-le-projet)
6. [Comprendre les résultats](#6-comprendre-les-résultats)
7. [Paramètres à ajuster](#7-paramètres-à-ajuster)
8. [Foire aux questions](#8-foire-aux-questions)

---

## 1. Vue d'ensemble

Ce projet implémente un **agent évolutionnaire** (algorithme génétique) qui apprend à jouer au Snake sur une grille dynamique. Il répond aux exigences du sujet 2 du module M2442.

**Principe général :**

```
Environnement Snake (grille dynamique)
        ↓ observations (12 features)
   Agent Neural (réseau 12→24→16→4)
        ↓ actions (Haut/Bas/Gauche/Droite)
   Algorithme Génétique (fait évoluer les poids)
        ↓ comparaison
   Baselines (aléatoire + heuristique)
```

**Ce que fait l'AG :** au lieu d'utiliser la rétropropagation, il fait évoluer une **population de réseaux de neurones** par sélection naturelle. Les meilleurs survivent et se reproduisent ; les moins bons disparaissent.

---

## 2. Installation

### Prérequis

- Python 3.8 ou supérieur
- pip

### Installer les dépendances

```bash
pip install gymnasium numpy
```

### Vérifier l'installation

```bash
python snake_env.py      # doit afficher une grille ASCII et "✅ Environnement OK !"
python neural_agent.py   # doit afficher "genome_size=780" et "✅ NeuralAgent OK !"
python baseline_agent.py # doit afficher les scores des 2 baselines
```

---

## 3. Structure du projet

```
snake_project/
│
├── snake_env.py          # Environnement Gymnasium (la grille Snake)
├── neural_agent.py       # Réseau de neurones = génome de l'AG
├── baseline_agent.py     # Agents de référence + fonction d'évaluation
├── genetic_algorithm.py  # Algorithme génétique complet
├── main.py               # Script principal (entraînement + comparaison)
│
├── results.json          # Généré après main.py : métriques + historique
└── best_genome.npy       # Généré après main.py : poids du meilleur agent
```

---

## 4. Fichiers en détail

### 4.1 `snake_env.py`

**Rôle :** définit l'environnement de jeu compatible avec l'API Gymnasium.

**Ce qui rend la grille "dynamique" :** à chaque `reset()`, trois éléments changent aléatoirement :
- La **taille** de la grille (choisie dans `{8, 10, 12, 14}`)
- Le **placement des obstacles** (~5% des cellules)
- La **position initiale de la nourriture**

**Formalisation MDP :**

| Élément | Valeur |
|---|---|
| Espace d'observations | `Box(0, 1, shape=(12,))` |
| Espace d'actions | `Discrete(4)` — 0=Haut, 1=Bas, 2=Gauche, 3=Droite |
| Récompense nourriture | `+10.0` |
| Récompense survie | `+0.1` par pas de temps |
| Pénalité collision | `-20.0` |
| Condition Done | collision mur/obstacle/corps ou `max_steps` atteint |

**Les 12 observations :**

```
[0]  danger_haut      – la cellule en haut est mortelle ?
[1]  danger_bas
[2]  danger_gauche
[3]  danger_droite
[4]  nourriture_en_haut    – la nourriture est au-dessus de la tête ?
[5]  nourriture_en_bas
[6]  nourriture_à_gauche
[7]  nourriture_à_droite
[8]  direction_haut   – one-hot de la direction actuelle
[9]  direction_bas
[10] direction_gauche
[11] direction_droite
```

**Utilisation basique :**

```python
from snake_env import SnakeEnv

env = SnakeEnv(render_mode="ansi")
obs, info = env.reset(seed=42)
print(f"Grille : {info['grid_size']}x{info['grid_size']}")

obs, reward, terminated, truncated, info = env.step(action=3)  # aller à droite
env.render()
```

---

### 4.2 `neural_agent.py`

**Rôle :** définit le réseau de neurones dont les poids constituent le **génome** que l'AG fait évoluer.

**Architecture :**

```
Entrée (12)  →  Couche cachée 1 (24)  →  Couche cachée 2 (16)  →  Sortie (4)
              tanh                      tanh                      argmax
```

**Taille du génome :** 780 paramètres (poids + biais de chaque couche), stockés dans un vecteur numpy 1D.

**Calcul :**
- Couche 1 : 12×24 + 24 = 312 paramètres
- Couche 2 : 24×16 + 16 = 400 paramètres
- Couche 3 : 16×4  + 4  =  68 paramètres
- **Total : 780**

**Utilisation :**

```python
from neural_agent import NeuralAgent
import numpy as np

agent = NeuralAgent()                 # poids aléatoires
action = agent.act(obs)               # retourne 0, 1, 2 ou 3

# Créer un agent depuis un génome sauvegardé
genome = np.load("best_genome.npy")
best_agent = NeuralAgent(genome=genome)
```

---

### 4.3 `baseline_agent.py`

**Rôle :** fournit deux agents de référence pour comparer les performances de l'AG, et la fonction d'évaluation commune à tous les agents.

#### `RandomAgent`

Choisit une action **uniformément au hasard** à chaque pas. Sert de plancher absolu : tout agent entraîné doit faire mieux.

```python
from baseline_agent import RandomAgent
from snake_env import SnakeEnv

env = SnakeEnv()
agent = RandomAgent(env.action_space)
action = agent.act(obs)  # action aléatoire
```

**Performance typique :** score moyen ~0.07, survie ~10 steps.

#### `HeuristicAgent`

Stratégie en deux étapes à chaque pas :

1. **Filtrer** les actions qui causent une collision immédiate (en lisant `obs[0:4]`)
2. **Choisir** parmi les actions sûres celle qui correspond à la direction de la nourriture (en lisant `obs[4:8]`)

Si toutes les actions sont dangereuses, choisit aléatoirement (cas rare).

```python
from baseline_agent import HeuristicAgent

agent = HeuristicAgent()  # pas besoin d'action_space
action = agent.act(obs)
```

**Performance typique :** score moyen ~14.7, survie ~155 steps. C'est la baseline principale à battre.

#### `evaluate_agent()`

Fonction utilitaire qui évalue **n'importe quel agent** (RandomAgent, HeuristicAgent, NeuralAgent) sur un nombre fixé d'épisodes avec des graines reproductibles.

```python
from baseline_agent import evaluate_agent

metrics = evaluate_agent(agent, n_episodes=200, seed_start=5000)
print(metrics["mean_score"])    # score moyen
print(metrics["mean_survival"]) # survie moyenne en steps
print(metrics["max_score"])     # meilleur épisode
```

**Métriques retournées :**

| Clé | Description |
|---|---|
| `scores` | liste brute des scores par épisode |
| `survivals` | liste brute des durées de survie |
| `mean_score` | moyenne des scores |
| `std_score` | écart-type des scores |
| `mean_survival` | durée de survie moyenne (steps) |
| `std_survival` | écart-type de la survie |
| `max_score` | meilleur score obtenu sur tous les épisodes |

---

### 4.4 `genetic_algorithm.py`

**Rôle :** implémente l'algorithme génétique complet qui fait évoluer une population de `NeuralAgent`.

#### Fonction de fitness

```
fitness = score × 100 + steps_survie × 0.5
```

Évaluée sur plusieurs épisodes (`n_eval`) pour réduire la variance liée à la grille dynamique.

#### Opérateurs génétiques

**Sélection — Tournoi :**
Tire `tournament_size` individus au hasard, garde le meilleur. Favorise les bons individus sans éliminer trop vite la diversité.

```python
def tournament_selection(population, fitnesses, tournament_size=3):
    idx = np.random.choice(len(population), size=tournament_size, replace=False)
    best_idx = idx[np.argmax([fitnesses[i] for i in idx])]
    return population[best_idx]
```

**Croisement — Uniforme :**
Chaque gène (poids) est hérité de l'un ou l'autre parent avec probabilité 50/50.

```python
def uniform_crossover(parent1, parent2):
    mask = np.random.rand(len(parent1.genome)) < 0.5
    child_genome = np.where(mask, parent1.genome, parent2.genome)
    return NeuralAgent(genome=child_genome)
```

**Mutation — Gaussienne adaptative :**
Chaque gène est muté avec probabilité `mutation_rate` en ajoutant du bruit `N(0, σ)`. Le `σ` décroît linéairement de `mutation_std_ini` à `mutation_std_end` au fil des générations (exploration forte au début, exploitation fine à la fin).

**Élitisme :**
Les `elite_size` meilleurs individus de chaque génération sont copiés directement dans la suivante, sans croisement ni mutation.

#### Utilisation

```python
from genetic_algorithm import GeneticAlgorithm

ag = GeneticAlgorithm(
    pop_size      = 80,
    n_generations = 50,
    elite_size    = 4,
    seed          = 42,
)
best_agent = ag.run()

# Accéder à l'historique d'apprentissage
print(ag.history["best_fitness"])  # fitness max par génération
print(ag.history["mean_score"])    # score moyen par génération
```

---

### 4.5 `main.py`

**Rôle :** script d'entrée qui orchestre tout le pipeline.

**Ce qu'il fait dans l'ordre :**

1. Lance l'entraînement AG avec les paramètres définis
2. Évalue les 3 agents sur 200 épisodes (graines identiques pour une comparaison équitable)
3. Affiche le tableau comparatif et la courbe d'apprentissage ASCII
4. Sauvegarde `results.json` et `best_genome.npy`

**Deux modes :**

```bash
python main.py          # entraînement complet (pop=80, 50 générations)
python main.py --fast   # version rapide pour tester (pop=30, 20 générations)
```

---

## 5. Lancer le projet

### Entraînement complet

```bash
python main.py
```

Durée estimée : 5 à 15 minutes selon la machine.

### Entraînement rapide (test)

```bash
python main.py --fast
```

Durée : moins d'une minute.

### Rejouer le meilleur agent

```python
import numpy as np
from neural_agent import NeuralAgent
from baseline_agent import evaluate_agent

genome = np.load("best_genome.npy")
best_agent = NeuralAgent(genome=genome)

metrics = evaluate_agent(best_agent, n_episodes=50, seed_start=9999)
print(f"Score moyen : {metrics['mean_score']:.2f}")
```

### Visualiser une partie en ASCII

```python
from snake_env import SnakeEnv
from neural_agent import NeuralAgent
import numpy as np, time

genome = np.load("best_genome.npy")
agent  = NeuralAgent(genome=genome)
env    = SnakeEnv(render_mode="ansi")

obs, _ = env.reset(seed=0)
done   = False
while not done:
    env.render()
    time.sleep(0.2)
    obs, _, terminated, truncated, _ = env.step(agent.act(obs))
    done = terminated or truncated
env.render()
```

---

## 6. Comprendre les résultats

### Tableau comparatif (200 épisodes, --fast mode)

| Agent | Score moyen | Survie moyenne | Max score |
|---|---|---|---|
| Aléatoire | 0.07 | 10.5 steps | 1 |
| Heuristique | 14.71 | 155.5 steps | 42 |
| AG (notre agent) | 0.51 | 55.9 steps | 3 |

> Note : ces résultats sont issus du mode `--fast` (20 générations, pop=30). L'entraînement complet (50 générations, pop=80) produit des résultats significativement meilleurs.

### Lire `results.json`

```json
{
  "params": { ... },          // paramètres utilisés
  "train_time_s": 3.72,       // durée d'entraînement
  "history": {
    "best_fitness": [...],    // fitness max à chaque génération
    "mean_fitness": [...],    // fitness moyenne à chaque génération
    "best_score":   [...]     // meilleur score (nourriture) à chaque génération
  },
  "metrics": {
    "random":    { ... },     // performances de l'agent aléatoire
    "heuristic": { ... },     // performances de l'heuristique
    "ag":        { ... }      // performances du meilleur agent AG
  }
}
```

---

## 7. Paramètres à ajuster

Les paramètres clés dans `main.py` et `GeneticAlgorithm` :

| Paramètre | Valeur par défaut | Effet |
|---|---|---|
| `pop_size` | 80 | Plus grand → meilleure exploration, plus lent |
| `n_generations` | 50 | Plus grand → plus de temps d'évolution |
| `elite_size` | 4 | Nombre de meilleurs individus conservés directement |
| `tournament_size` | 5 | Pression de sélection (3=faible, 7=forte) |
| `mutation_rate` | 0.08 | Proportion de gènes mutés par enfant |
| `mutation_std_ini` | 0.5 | Amplitude initiale des mutations (exploration) |
| `mutation_std_end` | 0.05 | Amplitude finale des mutations (exploitation) |
| `n_eval` | 5 | Épisodes par évaluation de fitness (↑ = plus stable, plus lent) |

Pour la grille dans `SnakeEnv` :

| Paramètre | Valeur par défaut | Effet |
|---|---|---|
| `grid_sizes` | `(8,10,12,14)` | Tailles possibles de la grille à chaque épisode |
| `obstacle_ratio` | `0.05` | Densité des obstacles (5% des cellules) |
| `max_steps_factor` | `3` | `max_steps = grid²×facteur` (évite les boucles) |

---

## 8. Foire aux questions

**Q : Pourquoi l'AG fait moins bien que l'heuristique en mode `--fast` ?**
L'AG a besoin de plus de générations et d'une population plus grande pour converger. Avec seulement 20 générations et 30 individus, il n'a pas assez de temps pour explorer l'espace des 780 paramètres. Utiliser `python main.py` (paramètres complets) donne de meilleurs résultats.

**Q : Pourquoi l'observation a-t-elle 12 features et pas plus ?**
12 features locales suffisent pour un réseau léger à évoluer par AG. Plus d'observations = génome plus grand = convergence plus lente. Un agent plus avancé pourrait utiliser toute la grille en entrée (CNN), mais ce serait trop coûteux pour un AG simple.

**Q : Comment reproduire exactement les mêmes résultats ?**
Utiliser `seed=42` dans `GeneticAlgorithm` et `seed_start=5000` dans `evaluate_agent` (déjà les valeurs par défaut dans `main.py`).

**Q : Comment sauvegarder et recharger le meilleur agent ?**
```python
import numpy as np
from neural_agent import NeuralAgent

# Sauvegarde (fait automatiquement par main.py)
np.save("best_genome.npy", best_agent.genome)

# Rechargement
genome = np.load("best_genome.npy")
agent  = NeuralAgent(genome=genome)
```

**Q : Peut-on changer l'architecture du réseau ?**
Oui, en modifiant `NeuralAgent.LAYER_SIZES` dans `neural_agent.py`. Attention : cela change la taille du génome, donc tout génome sauvegardé devient incompatible.

---

*Projet M2442 — Snake Évolutif sur Grille Dynamique*
