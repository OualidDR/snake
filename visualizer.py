"""
visualizer.py
=============
Visualiseur graphique du Snake avec Pygame.

Modes disponibles :
  python visualizer.py --agent ag         # meilleur agent AG (best_genome.npy)
  python visualizer.py --agent heuristic  # agent heuristique
  python visualizer.py --agent random     # agent aléatoire
  python visualizer.py --agent human      # joueur humain (flèches du clavier)

Options :
  --speed slow / normal / fast            # vitesse de défilement
  --episodes N                            # nombre de parties à jouer
"""

import sys
import os
import time
import argparse
import numpy as np

import pygame

from snake_env      import SnakeEnv, EMPTY, SNAKE, FOOD, OBSTACLE
from neural_agent   import NeuralAgent
from baseline_agent import RandomAgent, HeuristicAgent

# ─── Couleurs ────────────────────────────────────────────────────────────────
BG_COLOR       = ( 18,  18,  30)   # fond sombre
GRID_COLOR     = ( 35,  35,  55)   # lignes de grille
SNAKE_HEAD     = ( 80, 220, 120)   # tête serpent
SNAKE_BODY     = ( 50, 160,  90)   # corps serpent
FOOD_COLOR     = (255,  90,  90)   # nourriture
OBSTACLE_COLOR = ( 90,  90, 110)   # obstacle
TEXT_COLOR     = (210, 210, 230)   # texte principal
DIM_COLOR      = (120, 120, 150)   # texte secondaire
PANEL_COLOR    = ( 28,  28,  45)   # panneau latéral
ACCENT         = ( 80, 220, 120)   # accent vert

CELL_SIZE  = 40                    # pixels par cellule
PANEL_W    = 240                   # largeur du panneau d'info
MAX_GRID   = 14                    # taille max de la grille
WIN_W      = MAX_GRID * CELL_SIZE + PANEL_W
WIN_H      = MAX_GRID * CELL_SIZE + 60   # +60 pour la barre du bas

SPEEDS = {"slow": 4, "normal": 8, "fast": 20}

# ─── Argument parser ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent",    default="ag",
                   choices=["ag", "heuristic", "random", "human"])
    p.add_argument("--speed",    default="normal",
                   choices=["slow", "normal", "fast"])
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed",     type=int, default=None)
    return p.parse_args()


# ─── Helpers de dessin ───────────────────────────────────────────────────────
def draw_rounded_rect(surface, color, rect, radius=8):
    pygame.draw.rect(surface, color, rect, border_radius=radius)


def draw_cell(surface, grid_x, grid_y, color, offset_x, offset_y, shrink=2):
    x = offset_x + grid_x * CELL_SIZE + shrink
    y = offset_y + grid_y * CELL_SIZE + shrink
    s = CELL_SIZE - shrink * 2
    draw_rounded_rect(surface, color, (x, y, s, s), radius=6)


def draw_panel(surface, font_big, font_med, font_sm, info):
    """Dessine le panneau latéral avec les stats."""
    px = MAX_GRID * CELL_SIZE
    pygame.draw.rect(surface, PANEL_COLOR, (px, 0, PANEL_W, WIN_H))
    pygame.draw.line(surface, GRID_COLOR, (px, 0), (px, WIN_H), 2)

    y = 24
    # Titre
    title = font_big.render(" Snake AI", True, ACCENT)
    surface.blit(title, (px + 20, y)); y += 44

    # Séparateur
    pygame.draw.line(surface, GRID_COLOR, (px + 16, y), (px + PANEL_W - 16, y), 1)
    y += 16

    labels = [
        ("Agent",    info["agent_name"]),
        ("Épisode",  f"{info['episode']}"),
        ("Score",    f"{info['score']}  "),
        ("Steps",    f"{info['steps']}"),
        ("Grille",   f"{info['grid_size']}×{info['grid_size']}"),
        ("Meilleur", f"{info['best_score']}"),
    ]
    for label, value in labels:
        lbl_surf = font_sm.render(label, True, DIM_COLOR)
        val_surf = font_med.render(value, True, TEXT_COLOR)
        surface.blit(lbl_surf, (px + 20, y))
        surface.blit(val_surf, (px + 20, y + 18))
        y += 52

    y += 8
    pygame.draw.line(surface, GRID_COLOR, (px + 16, y), (px + PANEL_W - 16, y), 1)
    y += 16

    # Historique scores (mini-graphe)
    hist_label = font_sm.render("Historique scores", True, DIM_COLOR)
    surface.blit(hist_label, (px + 20, y)); y += 20

    scores = info.get("score_history", [])
    if scores:
        bar_area_w = PANEL_W - 40
        bar_h_max  = 60
        max_s = max(scores) if max(scores) > 0 else 1
        bar_w = max(4, bar_area_w // max(len(scores), 1))
        for i, s in enumerate(scores[-20:]):   # 20 derniers
            bh = int(s / max_s * bar_h_max)
            bx = px + 20 + i * bar_w
            by = y + bar_h_max - bh
            col = ACCENT if i == len(scores[-20:]) - 1 else (60, 140, 80)
            pygame.draw.rect(surface, col, (bx, by, max(bar_w - 2, 2), bh), border_radius=2)
        y += bar_h_max + 10

    # Contrôles
    y += 8
    pygame.draw.line(surface, GRID_COLOR, (px + 16, y), (px + PANEL_W - 16, y), 1)
    y += 14
    controls = [
        "ESPACE  pause/reprendre",
        "R       reset épisode",
        "+/-     vitesse",
        "ESC     quitter",
    ]
    for c in controls:
        s = font_sm.render(c, True, DIM_COLOR)
        surface.blit(s, (px + 20, y)); y += 18


def draw_grid(surface, env, offset_x, offset_y):
    """Dessine la grille du jeu."""
    gs = env.grid_size

    # Fond de la zone jeu
    pygame.draw.rect(surface, BG_COLOR,
                     (offset_x, offset_y, gs * CELL_SIZE, gs * CELL_SIZE))

    # Lignes de grille
    for i in range(gs + 1):
        pygame.draw.line(surface, GRID_COLOR,
                         (offset_x + i * CELL_SIZE, offset_y),
                         (offset_x + i * CELL_SIZE, offset_y + gs * CELL_SIZE))
        pygame.draw.line(surface, GRID_COLOR,
                         (offset_x, offset_y + i * CELL_SIZE),
                         (offset_x + gs * CELL_SIZE, offset_y + i * CELL_SIZE))

    # Cellules
    for r in range(gs):
        for c in range(gs):
            cell = env.grid[r, c]
            if cell == OBSTACLE:
                draw_cell(surface, c, r, OBSTACLE_COLOR, offset_x, offset_y, shrink=3)
            elif cell == FOOD:
                # Cercle pour la nourriture
                cx = offset_x + c * CELL_SIZE + CELL_SIZE // 2
                cy = offset_y + r * CELL_SIZE + CELL_SIZE // 2
                pygame.draw.circle(surface, FOOD_COLOR, (cx, cy), CELL_SIZE // 2 - 5)
            elif cell == SNAKE:
                is_head = ((r, c) == env.snake[0])
                col = SNAKE_HEAD if is_head else SNAKE_BODY
                draw_cell(surface, c, r, col, offset_x, offset_y, shrink=2 if is_head else 4)


def draw_status_bar(surface, font_sm, fps, paused, offset_y):
    """Barre de statut en bas."""
    pygame.draw.rect(surface, PANEL_COLOR,
                     (0, offset_y, MAX_GRID * CELL_SIZE + PANEL_W, 60))
    pygame.draw.line(surface, GRID_COLOR, (0, offset_y),
                     (MAX_GRID * CELL_SIZE + PANEL_W, offset_y), 1)
    status = "⏸ PAUSE — appuie sur ESPACE" if paused else f"▶ En cours  |  FPS: {fps:.0f}"
    s = font_sm.render(status, True, DIM_COLOR)
    surface.blit(s, (20, offset_y + 18))


def draw_episode_end(surface, font_big, font_med, score, best):
    """Overlay de fin d'épisode."""
    overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surface.blit(overlay, (0, 0))

    msg1 = font_big.render("Episode terminé", True, TEXT_COLOR)
    msg2 = font_med.render(f"Score : {score}   |   Meilleur : {best}", True, ACCENT)
    msg3 = font_med.render("Appuie sur R pour rejouer  /  ESC pour quitter", True, DIM_COLOR)

    surface.blit(msg1, (WIN_W // 2 - msg1.get_width() // 2, WIN_H // 2 - 70))
    surface.blit(msg2, (WIN_W // 2 - msg2.get_width() // 2, WIN_H // 2 - 10))
    surface.blit(msg3, (WIN_W // 2 - msg3.get_width() // 2, WIN_H // 2 + 40))


# ─── Boucle principale ───────────────────────────────────────────────────────
def run(args):
    pygame.init()
    pygame.display.set_caption("Snake Évolutif M2442")

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock  = pygame.time.Clock()

    # Polices
    font_big = pygame.font.SysFont("segoeui", 26, bold=True)
    font_med = pygame.font.SysFont("segoeui", 20)
    font_sm  = pygame.font.SysFont("segoeui", 15)

    # Charger l'agent
    agent_name = args.agent
    env        = SnakeEnv()

    if agent_name == "ag":
        genome_path = "best_genome.npy"
        if not os.path.exists(genome_path):
            print("best_genome.npy introuvable. Lance d'abord : python main.py")
            pygame.quit(); return
        genome = np.load(genome_path)
        agent  = NeuralAgent(genome=genome)
        agent_name = "AG (évolué)"
    elif agent_name == "heuristic":
        agent      = HeuristicAgent()
        agent_name = "Heuristique"
    elif agent_name == "random":
        agent      = RandomAgent(env.action_space)
        agent_name = "Aléatoire"
    else:
        agent      = None
        agent_name = "Humain 🎮"

    # Décalage pour centrer la grille (taille max)
    offset_x = (MAX_GRID * CELL_SIZE - MAX_GRID * CELL_SIZE) // 2
    offset_y = 0

    # État
    fps_target   = SPEEDS[args.speed]
    episode      = 0
    best_score   = 0
    score_history = []
    paused       = False
    episode_done = False
    waiting      = False   # attente après fin d'épisode

    seed = args.seed if args.seed is not None else 0
    obs, _ = env.reset(seed=seed)
    episode = 1

    running = True
    while running:
        dt = clock.tick(fps_target)

        # ── Événements ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r or (waiting and event.key != pygame.K_ESCAPE):
                    # Reset
                    seed += 1
                    obs, _ = env.reset(seed=seed)
                    episode     += 1
                    episode_done = False
                    waiting      = False
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    fps_target = min(fps_target + 2, 60)
                elif event.key == pygame.K_MINUS:
                    fps_target = max(fps_target - 2, 1)

                # Contrôle humain
                elif agent is None and not paused and not episode_done:
                    key_to_action = {
                        pygame.K_UP: 0, pygame.K_DOWN: 1,
                        pygame.K_LEFT: 2, pygame.K_RIGHT: 3,
                        pygame.K_z: 0, pygame.K_s: 1,
                        pygame.K_q: 2, pygame.K_d: 3,
                    }
                    if event.key in key_to_action:
                        action = key_to_action[event.key]
                        obs, _, terminated, truncated, info = env.step(action)
                        episode_done = terminated or truncated
                        if episode_done:
                            best_score = max(best_score, info.get("score", 0))
                            score_history.append(info.get("score", 0))
                            waiting = True

        # ── Logique agent (non-humain) ──
        if not paused and not episode_done and agent is not None:
            action = agent.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            episode_done = terminated or truncated

            if episode_done:
                best_score = max(best_score, info.get("score", 0))
                score_history.append(info.get("score", 0))
                waiting = True
                # Auto-reset après une pause visuelle
                time.sleep(0.6)
                if episode < args.episodes:
                    seed += 1
                    obs, _ = env.reset(seed=seed)
                    episode     += 1
                    episode_done = False
                    waiting      = False

        # ── Dessin ──
        screen.fill(BG_COLOR)

        # Zone jeu (fond étendu si grille < max)
        pygame.draw.rect(screen, BG_COLOR,
                         (0, 0, MAX_GRID * CELL_SIZE, WIN_H - 60))

        draw_grid(screen, env, offset_x, offset_y)

        panel_info = {
            "agent_name"   : agent_name,
            "episode"      : episode,
            "score"        : env.score,
            "steps"        : env.step_count,
            "grid_size"    : env.grid_size,
            "best_score"   : best_score,
            "score_history": score_history,
        }
        draw_panel(screen, font_big, font_med, font_sm, panel_info)
        draw_status_bar(screen, font_sm, clock.get_fps(), paused, WIN_H - 60)

        if waiting:
            draw_episode_end(screen, font_big, font_med, env.score, best_score)

        pygame.display.flip()

        # Quitter si épisodes terminés
        if waiting and episode >= args.episodes and agent is not None:
            time.sleep(1.5)
            running = False

    env.close()
    pygame.quit()


# ─── Entrée ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    run(args)
