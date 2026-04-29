"""
Pygame visualizer for Death Maze RL.
Renders the maze, agent, and enemies in real-time.
"""

import sys
import pygame
import numpy as np
from maze_env import DeathMazeEnv, WALL, EMPTY, START, GOAL, TRAP, POISON, ENEMY, MAX_HEALTH

CELL        = 44
INFO_H      = 90
BORDER      = 1

PALETTE = {
    WALL:   (38,  38,  42),
    EMPTY:  (225, 223, 215),
    START:  (72,  180, 80),
    GOAL:   (255, 210, 0),
    TRAP:   (215, 48,  48),
    POISON: (148, 28,  185),
    ENEMY:  (30,  120, 225),
}
AGENT_COLOR   = (255, 95,  30)
AGENT_RING    = (255, 185, 120)
BG_COLOR      = (16,  16,  20)
HUD_COLOR     = (26,  26,  32)
TEXT_COLOR    = (220, 220, 220)
DIM_COLOR     = (110, 110, 110)
HP_HIGH       = (70,  210, 70)
HP_LOW        = (215, 55,  55)
OUTCOME_COLORS = {
    "goal":    (70,  220, 70),
    "dead":    (220, 55,  55),
    "timeout": (200, 180, 40),
}


class MazeVisualizer:
    def __init__(self, env: DeathMazeEnv, fps: int = 10):
        self.env = env
        self.fps = fps

        pygame.init()
        pygame.font.init()

        self.W_px = env.W * CELL
        self.H_px = env.H * CELL + INFO_H
        self.screen = pygame.display.set_mode((self.W_px, self.H_px))
        pygame.display.set_caption("Death Maze RL")

        self.font_sm = pygame.font.SysFont("monospace", 14)
        self.font_md = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_lg = pygame.font.SysFont("monospace", 32, bold=True)
        self.clock   = pygame.time.Clock()

    # ------------------------------------------------------------------
    def _draw_grid(self):
        env = self.env
        grid = env.base_grid.copy()
        for e in env.enemies:
            er, ec = e.pos
            grid[er, ec] = ENEMY

        for r in range(env.H):
            for c in range(env.W):
                tile  = int(grid[r, c])
                color = PALETTE.get(tile, (200, 200, 200))
                rect  = pygame.Rect(c * CELL, r * CELL, CELL - BORDER, CELL - BORDER)
                pygame.draw.rect(self.screen, color, rect, border_radius=4)

                cx = c * CELL + CELL // 2
                cy = r * CELL + CELL // 2

                if tile == GOAL:
                    label = self.font_md.render("G", True, (80, 60, 0))
                    self.screen.blit(label, label.get_rect(center=(cx, cy)))
                elif tile == TRAP:
                    label = self.font_sm.render("T", True, (255, 190, 190))
                    self.screen.blit(label, label.get_rect(center=(cx, cy)))
                elif tile == POISON:
                    label = self.font_sm.render("P", True, (215, 175, 255))
                    self.screen.blit(label, label.get_rect(center=(cx, cy)))
                elif tile == START:
                    label = self.font_sm.render("S", True, (30, 80, 35))
                    self.screen.blit(label, label.get_rect(center=(cx, cy)))

    def _draw_enemies(self):
        for e in self.env.enemies:
            er, ec = e.pos
            cx = ec * CELL + CELL // 2
            cy = er * CELL + CELL // 2
            r  = CELL // 2 - 5
            pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
            pygame.draw.polygon(self.screen, PALETTE[ENEMY], pts)
            pygame.draw.polygon(self.screen, (10, 55, 160), pts, 2)
            label = self.font_sm.render("E", True, (200, 230, 255))
            self.screen.blit(label, label.get_rect(center=(cx, cy)))

    def _draw_agent(self):
        r, c = self.env.agent_pos
        cx = c * CELL + CELL // 2
        cy = r * CELL + CELL // 2
        rad = CELL // 2 - 5
        pygame.draw.circle(self.screen, AGENT_COLOR, (cx, cy), rad)
        pygame.draw.circle(self.screen, AGENT_RING,  (cx, cy), rad, 2)

    def _draw_hud(self, episode: int, step: int, total_reward: float, outcome: str = None):
        hud_y = self.env.H * CELL
        pygame.draw.rect(self.screen, HUD_COLOR,
                         pygame.Rect(0, hud_y, self.W_px, INFO_H))

        # Health bar
        hp_frac = max(0.0, self.env.health / MAX_HEALTH)
        bar_x, bar_y, bar_w, bar_h = 10, hud_y + 8, self.W_px - 20, 13
        pygame.draw.rect(self.screen, (55, 55, 55),
                         pygame.Rect(bar_x, bar_y, bar_w, bar_h), border_radius=4)
        hp_color = HP_HIGH if hp_frac > 0.4 else HP_LOW
        pygame.draw.rect(self.screen, hp_color,
                         pygame.Rect(bar_x, bar_y, int(bar_w * hp_frac), bar_h), border_radius=4)
        hp_txt = self.font_sm.render(f"HP {self.env.health}/{MAX_HEALTH}", True, TEXT_COLOR)
        self.screen.blit(hp_txt, (bar_x + 4, bar_y - 1))

        # Stats line
        stats = self.font_sm.render(
            f"Episode: {episode:4d}   Step: {step:3d}   Reward: {total_reward:+7.0f}",
            True, TEXT_COLOR,
        )
        self.screen.blit(stats, (10, hud_y + 30))

        # Outcome banner
        if outcome:
            label_map = {"goal": "GOAL REACHED!", "dead": "AGENT DIED", "timeout": "TIMEOUT"}
            banner = self.font_md.render(label_map.get(outcome, outcome),
                                         True, OUTCOME_COLORS.get(outcome, TEXT_COLOR))
            self.screen.blit(banner, (10, hud_y + 56))
        else:
            hint = self.font_sm.render("ESC to quit", True, DIM_COLOR)
            self.screen.blit(hint, (10, hud_y + 58))

    # ------------------------------------------------------------------
    def _pump(self):
        """Handle quit events; return False if user wants to exit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def render_frame(self, episode: int, step: int, total_reward: float, outcome: str = None) -> bool:
        if not self._pump():
            return False
        self.screen.fill(BG_COLOR)
        self._draw_grid()
        self._draw_enemies()
        self._draw_agent()
        self._draw_hud(episode, step, total_reward, outcome)
        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    # ------------------------------------------------------------------
    def play_episode(self, agent, episode: int, fps: int = None, training: bool = True) -> str:
        """Run one episode with full rendering. Returns outcome string.

        training=True  uses epsilon-greedy (shows real behaviour at that point).
        training=False forces greedy (useful for a final polished demo).
        """
        saved_fps = self.fps
        if fps is not None:
            self.fps = fps

        obs, _ = self.env.reset()
        total_reward = 0.0
        step = 0
        outcome = "timeout"

        while True:
            if not self.render_frame(episode, step, total_reward):
                self.fps = saved_fps
                return outcome

            action = agent.select_action(obs, training=training)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            step += 1

            if terminated or truncated:
                outcome = info.get("outcome", "timeout")
                self.render_frame(episode, step, total_reward, outcome)
                pygame.time.wait(2000)
                break

        self.fps = saved_fps
        return outcome

    # ------------------------------------------------------------------
    def show_training_screen(self, ep: int, total_ep: int, avg_reward: float, epsilon: float):
        """Lightweight progress overlay shown between demos during training."""
        if not self._pump():
            pygame.quit()
            sys.exit()

        self.screen.fill((12, 12, 18))

        cx = self.W_px // 2

        title = self.font_lg.render("TRAINING IN PROGRESS", True, (90, 140, 255))
        self.screen.blit(title, title.get_rect(center=(cx, self.H_px // 2 - 90)))

        # Progress bar
        bar_w = self.W_px - 100
        bar_x = 50
        bar_y = self.H_px // 2 - 40
        frac  = ep / total_ep
        pygame.draw.rect(self.screen, (40, 40, 50),
                         pygame.Rect(bar_x, bar_y, bar_w, 18), border_radius=5)
        pygame.draw.rect(self.screen, (90, 140, 255),
                         pygame.Rect(bar_x, bar_y, int(bar_w * frac), 18), border_radius=5)

        pct = self.font_md.render(
            f"Episode {ep} / {total_ep}  ({frac*100:.0f}%)", True, (200, 200, 200)
        )
        self.screen.blit(pct, pct.get_rect(center=(cx, self.H_px // 2 + 5)))

        stats = self.font_sm.render(
            f"Avg reward (last 100): {avg_reward:+.1f}   Epsilon: {epsilon:.3f}",
            True, (150, 150, 150),
        )
        self.screen.blit(stats, stats.get_rect(center=(cx, self.H_px // 2 + 35)))

        hint = self.font_sm.render("ESC to quit", True, (70, 70, 70))
        self.screen.blit(hint, hint.get_rect(center=(cx, self.H_px // 2 + 65)))

        pygame.display.flip()

    # ------------------------------------------------------------------
    def close(self):
        pygame.quit()
