"""
snake_env.py
============
Environnement Gymnasium personnalisé pour le Snake évolutif sur grille dynamique.

MDP Formel :
  - États (observations) : vecteur de 12 features décrivant l'état local du serpent
  - Actions             : 0=Haut, 1=Bas, 2=Gauche, 3=Droite (discret)
  - Récompenses         : +10 nourriture, +0.1/pas survie, -20 collision, -0.5 demi-tour
  - Done                : collision mur/obstacle/soi-même, ou dépassement max_steps

Grille Dynamique (varie à chaque reset) :
  - Taille de la grille : choisie aléatoirement dans [8, 10, 12, 14]
  - Nombre d'obstacles  : proportionnel à la taille (5% des cellules)
  - Placement aléatoire des obstacles et de la nourriture à chaque épisode
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# ─────────────────────────────────────────────
#  Constantes de représentation de la grille
# ─────────────────────────────────────────────
EMPTY    = 0
SNAKE    = 1
FOOD     = 2
OBSTACLE = 3

# Correspondance action → vecteur direction (row, col)
ACTION_TO_DIR = {
    0: (-1,  0),  # Haut
    1: ( 1,  0),  # Bas
    2: ( 0, -1),  # Gauche
    3: ( 0,  1),  # Droite
}

OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}  # action opposée


class SnakeEnv(gym.Env):
    """
    Snake évolutif sur grille dynamique.
    Compatible avec l'API Gymnasium standard (reset / step / render).
    """

    metadata = {"render_modes": ["ansi"]}

    # ──────────────────────────────────────────
    def __init__(self, render_mode=None,
                 grid_sizes=(8, 10, 12, 14),
                 obstacle_ratio=0.05,
                 max_steps_factor=3):
        """
        Paramètres
        ----------
        grid_sizes       : tuple des tailles possibles (choisie aléatoirement au reset)
        obstacle_ratio   : fraction des cellules occupées par des obstacles
        max_steps_factor : max_steps = grid_size² × facteur  (évite les boucles infinies)
        """
        super().__init__()
        self.render_mode      = render_mode
        self.grid_sizes       = grid_sizes
        self.obstacle_ratio   = obstacle_ratio
        self.max_steps_factor = max_steps_factor

        # ── Espaces (dimension fixée sur la plus grande grille possible) ──
        # Observation : 12 features binaires/normalisées (voir _get_obs)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )
        # Action : discret 4 directions
        self.action_space = spaces.Discrete(4)

        # État interne (initialisé au reset)
        self.grid        = None
        self.snake       = []   # liste de (row, col), tête en [0]
        self.direction   = None
        self.food_pos    = None
        self.step_count  = 0
        self.max_steps   = 0
        self.grid_size   = None
        self.score       = 0

    # ──────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # ── 1. Choisir la taille de la grille dynamiquement ──
        self.grid_size = int(self.np_random.choice(self.grid_sizes))
        self.max_steps = self.grid_size ** 2 * self.max_steps_factor

        # ── 2. Grille vide ──
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # ── 3. Placer les obstacles aléatoirement ──
        n_obstacles = max(1, int(self.grid_size ** 2 * self.obstacle_ratio))
        obstacle_cells = self._random_empty_cells(n_obstacles)
        for r, c in obstacle_cells:
            self.grid[r, c] = OBSTACLE

        # ── 4. Placer le serpent (3 cellules) au centre ──
        center = self.grid_size // 2
        self.snake     = [(center, center),
                          (center, center + 1),
                          (center, center + 2)]
        self.direction = 2  # Gauche (s'éloigne de la queue)
        for r, c in self.snake:
            self.grid[r, c] = SNAKE

        # ── 5. Placer la nourriture ──
        self.food_pos = self._place_food()
        self.grid[self.food_pos] = FOOD

        self.step_count = 0
        self.score      = 0

        obs = self._get_obs()
        info = {"grid_size": self.grid_size}
        return obs, info

    # ──────────────────────────────────────────
    def step(self, action):
        # Ignorer le demi-tour (empêche la mort instantanée par mauvaise action)
        if action == OPPOSITE[self.direction]:
            action = self.direction

        self.direction  = action
        dr, dc          = ACTION_TO_DIR[action]
        head_r, head_c  = self.snake[0]
        new_r, new_c    = head_r + dr, head_c + dc

        reward      = 0.1   # récompense de survie à chaque pas
        terminated  = False
        truncated   = False

        # ── Collision mur ──
        if not (0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size):
            terminated = True
            reward     = -20.0
            return self._get_obs(), reward, terminated, truncated, {"score": self.score}

        cell = self.grid[new_r, new_c]

        # ── Collision obstacle ou corps ──
        if cell == OBSTACLE or cell == SNAKE:
            terminated = True
            reward     = -20.0
            return self._get_obs(), reward, terminated, truncated, {"score": self.score}

        # ── Nourriture ──
        ate_food = (cell == FOOD)
        if ate_food:
            reward       = 10.0
            self.score  += 1
        else:
            # Enlever la queue si pas de nourriture
            tail_r, tail_c = self.snake[-1]
            self.snake.pop()
            self.grid[tail_r, tail_c] = EMPTY

        # Avancer la tête
        self.snake.insert(0, (new_r, new_c))
        self.grid[new_r, new_c] = SNAKE

        # Placer nouvelle nourriture si mangée
        if ate_food:
            new_food = self._place_food()
            if new_food is None:
                # Grille pleine → victoire
                terminated = True
            else:
                self.food_pos            = new_food
                self.grid[new_food]      = FOOD

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        obs  = self._get_obs()
        info = {"score": self.score, "steps": self.step_count}
        return obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────
    def _get_obs(self):
        """
        Observation : vecteur de 12 features (toutes dans [0, 1])

        [0]  danger_devant    – la prochaine cellule dans la direction actuelle est mortelle
        [1]  danger_gauche    – danger si on tourne à gauche
        [2]  danger_droite    – danger si on tourne à droite
        [3]  danger_derriere  – (toujours 0 ou ignoré, mais utile pour le réseau)

        [4]  nourriture_haut
        [5]  nourriture_bas
        [6]  nourriture_gauche
        [7]  nourriture_droite

        [8]  direction_haut   (one-hot)
        [9]  direction_bas
        [10] direction_gauche
        [11] direction_droite
        """
        head_r, head_c = self.snake[0] if self.snake else (0, 0)

        def is_deadly(r, c):
            if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
                return 1.0
            if self.grid[r, c] == OBSTACLE or self.grid[r, c] == SNAKE:
                return 1.0
            return 0.0

        # Dangers dans les 4 directions absolues
        dangers = {}
        for a, (dr, dc) in ACTION_TO_DIR.items():
            dangers[a] = is_deadly(head_r + dr, head_c + dc)

        # Nourriture relative
        food_r, food_c = self.food_pos if self.food_pos else (head_r, head_c)
        food_up    = float(food_r < head_r)
        food_down  = float(food_r > head_r)
        food_left  = float(food_c < head_c)
        food_right = float(food_c > head_c)

        # Direction actuelle (one-hot)
        dir_oh = [0.0] * 4
        dir_oh[self.direction] = 1.0

        obs = np.array([
            dangers[0], dangers[1], dangers[2], dangers[3],
            food_up, food_down, food_left, food_right,
            dir_oh[0], dir_oh[1], dir_oh[2], dir_oh[3],
        ], dtype=np.float32)

        return obs

    # ──────────────────────────────────────────
    def render(self):
        if self.render_mode != "ansi":
            return
        symbols = {EMPTY: "·", SNAKE: "O", FOOD: "F", OBSTACLE: "#"}
        lines   = []
        border  = "+" + "-" * (self.grid_size * 2 - 1) + "+"
        lines.append(border)
        for r in range(self.grid_size):
            row_str = "|"
            for c in range(self.grid_size):
                cell = self.grid[r, c]
                # Marquer la tête différemment
                if (r, c) == self.snake[0]:
                    row_str += "H"
                else:
                    row_str += symbols[cell]
                if c < self.grid_size - 1:
                    row_str += " "
            row_str += "|"
            lines.append(row_str)
        lines.append(border)
        lines.append(f"Score: {self.score}  |  Steps: {self.step_count}  |  Grid: {self.grid_size}x{self.grid_size}")
        print("\n".join(lines))

    # ──────────────────────────────────────────
    def _random_empty_cells(self, n):
        """Retourne n positions vides aléatoires sur la grille."""
        empty = [(r, c)
                 for r in range(self.grid_size)
                 for c in range(self.grid_size)
                 if self.grid[r, c] == EMPTY]
        self.np_random.shuffle(empty)
        return empty[:n]

    def _place_food(self):
        """Place la nourriture sur une cellule vide aléatoire. Retourne None si aucune."""
        candidates = self._random_empty_cells(1)
        return candidates[0] if candidates else None

    def close(self):
        pass
