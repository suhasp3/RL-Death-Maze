"""
Death Maze RL - Custom Gym Environment
A 15x15 grid maze with traps, poison tiles, and patrolling enemies.
"""

import numpy as np
import random

# Minimal Gym-compatible base (no gymnasium required)
class _Env:
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        return None, {}
    def step(self, action): ...
    def render(self): ...
    def close(self): ...

class _Space:
    pass

class _Box(_Space):
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

class _Discrete(_Space):
    def __init__(self, n):
        self.n = n
    def sample(self):
        return np.random.randint(self.n)

# Tile types
EMPTY   = 0
WALL    = 1
START   = 2
GOAL    = 3
TRAP    = 4
POISON  = 5
ENEMY   = 6  # rendered position only

# Actions
UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3
WAIT  = 4

ACTION_DELTAS = {
    UP:    (-1,  0),
    DOWN:  ( 1,  0),
    LEFT:  ( 0, -1),
    RIGHT: ( 0,  1),
    WAIT:  ( 0,  0),
}

# Rewards
R_GOAL        =  100
R_STEP        =   -1
R_WALL        =   -3
R_TRAP        =  -20
R_POISON      =   -5
R_DEATH       = -100
R_CLOSER      =  0.5
R_FARTHER     = -0.5

MAX_HEALTH    = 100
POISON_DAMAGE =  10
MAX_STEPS     = 300

# Fixed 15x15 maze layout
# '#' = wall, '.' = empty, 'S' = start, 'G' = goal,
# 'T' = trap, 'P' = poison, 'M' = enemy patrol start
MAZE_TEMPLATE = [
    "#################",
    "#S..#...T...#..G#",
    "#.#.#.#####.#.###",
    "#.#...#...#.....#",
    "#.###.#.#.#####.#",
    "#.#P..#.#.....#.#",
    "#.#.###.#####.#.#",
    "#...#.....#...#.#",
    "#.###.###.#.###.#",
    "#.#T..#.#...#...#",
    "#.#.#.#.#####.#.#",
    "#...#.P.......#.#",
    "#.#####.#####.#.#",
    "#.......#M....#.#",
    "#################",
]

# Patrol paths for enemies (list of (row, col) waypoints)
PATROL_PATHS = [
    [(13, 9), (13, 10), (13, 11), (13, 12)],   # Enemy 0: bottom horizontal
    [(5, 1),  (6, 1),  (7, 1),  (8, 1),  (9, 1), (10, 1), (11, 1)],  # Enemy 1: left corridor vertical
    [(7, 5),  (7, 6),  (7, 7),  (7, 8),  (7, 9)],  # Enemy 2: middle horizontal
]


def parse_maze(template):
    """Convert string template to numpy grid and extract special positions."""
    char_map = {
        '#': WALL, '.': EMPTY, 'S': START, 'G': GOAL,
        'T': TRAP,  'P': POISON, 'M': EMPTY,  # M tile is walkable, enemy overlaid
    }
    rows = []
    for line in template:
        row = [char_map[c] for c in line]
        rows.append(row)
    grid = np.array(rows, dtype=np.int32)
    return grid


class PatrolEnemy:
    """An enemy that walks a fixed waypoint loop."""

    def __init__(self, path):
        self.path = path
        self.idx = 0
        self.direction = 1  # 1 = forward, -1 = backward (ping-pong)

    @property
    def pos(self):
        return self.path[self.idx]

    def step(self):
        next_idx = self.idx + self.direction
        if next_idx >= len(self.path) or next_idx < 0:
            self.direction *= -1
            next_idx = self.idx + self.direction
        self.idx = next_idx

    def reset(self):
        self.idx = 0
        self.direction = 1


class DeathMazeEnv(_Env):
    """
    Death Maze RL Environment.

    Observation: flat vector of [agent_row, agent_col, health,
                                  local 5x5 patch (25 values),
                                  dist_to_goal]
    Action: Discrete(5) — UP, DOWN, LEFT, RIGHT, WAIT
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.base_grid = parse_maze(MAZE_TEMPLATE)
        self.H, self.W = self.base_grid.shape

        # Find start and goal
        self.start_pos = tuple(zip(*np.where(self.base_grid == START)))[0]
        self.goal_pos  = tuple(zip(*np.where(self.base_grid == GOAL)))[0]
        self.start_pos = (int(self.start_pos[0]), int(self.start_pos[1]))
        self.goal_pos  = (int(self.goal_pos[0]),  int(self.goal_pos[1]))

        # Enemies
        self.enemies = [PatrolEnemy(path) for path in PATROL_PATHS]

        # Observation: 3 scalars + 25 local patch + 1 dist = 29
        obs_size = 3 + 25 + 1
        self.observation_space = _Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = _Discrete(5)

        self.agent_pos = self.start_pos
        self.health = MAX_HEALTH
        self.steps = 0
        self.done = False

    # ------------------------------------------------------------------
    def _get_obs(self):
        r, c = self.agent_pos
        # Build current grid with enemies overlaid
        grid = self.base_grid.copy()
        for e in self.enemies:
            er, ec = e.pos
            grid[er, ec] = ENEMY

        # 5x5 local patch (padded with walls outside bounds)
        patch = np.ones((5, 5), dtype=np.float32)  # default = wall = 1
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.H and 0 <= nc < self.W:
                    patch[dr + 2, dc + 2] = grid[nr, nc] / 6.0  # normalize

        dist = (abs(r - self.goal_pos[0]) + abs(c - self.goal_pos[1])) / (self.H + self.W)

        obs = np.array([
            r / self.H,
            c / self.W,
            self.health / MAX_HEALTH,
            *patch.flatten(),
            dist,
        ], dtype=np.float32)
        return obs

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        self.health = MAX_HEALTH
        self.steps = 0
        self.done = False
        for e in self.enemies:
            e.reset()
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        assert not self.done, "Call reset() before stepping."
        self.steps += 1
        reward = R_STEP
        terminated = False
        truncated = False
        info = {}

        prev_dist = self._manhattan(self.agent_pos, self.goal_pos)

        # --- Move agent ---
        dr, dc = ACTION_DELTAS[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        if action == WAIT:
            pass  # stay in place, still pay step cost
        elif not (0 <= nr < self.H and 0 <= nc < self.W) or self.base_grid[nr, nc] == WALL:
            reward += R_WALL
        else:
            self.agent_pos = (nr, nc)

        # --- Shaping: distance reward ---
        new_dist = self._manhattan(self.agent_pos, self.goal_pos)
        if new_dist < prev_dist:
            reward += R_CLOSER
        elif new_dist > prev_dist:
            reward += R_FARTHER

        # --- Check tile effects ---
        tile = self.base_grid[self.agent_pos[0], self.agent_pos[1]]

        if tile == GOAL:
            reward += R_GOAL
            terminated = True
            info["outcome"] = "goal"

        elif tile == TRAP:
            reward += R_TRAP
            self.health -= 30
            info["hit_trap"] = True

        elif tile == POISON:
            reward += R_POISON
            self.health -= POISON_DAMAGE
            info["hit_poison"] = True

        # --- Move enemies ---
        for e in self.enemies:
            e.step()
            if e.pos == self.agent_pos:
                reward += R_DEATH
                self.health = 0
                info["killed_by_enemy"] = True

        # --- Death check ---
        if self.health <= 0:
            reward += R_DEATH
            terminated = True
            info["outcome"] = "dead"

        # --- Timeout ---
        if self.steps >= MAX_STEPS:
            truncated = True
            info["outcome"] = "timeout"

        self.done = terminated or truncated
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        grid = [list(row) for row in MAZE_TEMPLATE]
        # Place enemies
        for e in self.enemies:
            er, ec = e.pos
            grid[er][ec] = 'M'
        # Place agent
        r, c = self.agent_pos
        grid[r][c] = 'A'
        print(f"Steps: {self.steps}  Health: {self.health}")
        print('\n'.join(''.join(row) for row in grid))
        print()
