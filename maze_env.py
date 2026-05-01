"""
Death Maze RL - Custom Gym Environment (25x25)
Sparse Reward Version: Agent must discover the Plate-Trap connection 
without a direct reward signal for the plate.
"""

import numpy as np
import random

# -----------------------------------------------------------------------
# Gym-compatible base classes
# -----------------------------------------------------------------------
class _Env:
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        return None, {}
    def step(self, action): ...
    def render(self): ...
    def close(self): ...

class _Box:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

class _Discrete:
    def __init__(self, n):
        self.n = n
    def sample(self):
        return np.random.randint(self.n)

# -----------------------------------------------------------------------
# Constants & Tile Types
# -----------------------------------------------------------------------
EMPTY    = 0
WALL     = 1
START    = 2
GOAL     = 3
TRAP     = 4
POISON   = 5
ENEMY    = 6
PLATE    = 7  
D_TRAP   = 8  

UP, DOWN, LEFT, RIGHT, WAIT = 0, 1, 2, 3, 4

ACTION_DELTAS = {
    UP:    (-1,  0),
    DOWN:  ( 1,  0),
    LEFT:  ( 0, -1),
    RIGHT: ( 0,  1),
    WAIT:  ( 0,  0),
}

# Rewards (REMOVED R_PLATE)
R_GOAL    = 1000
R_STEP    = -0.5
R_WALL    = -4
R_TRAP    = -50
R_POISON  = -5
R_DEATH   = -300
R_CLOSER  =  0.15
R_FARTHER = -0.15
R_EXPLORE =  0.75  # bonus for visiting a cell for the first time this episode

MAX_HEALTH    = 100
TRAP_DAMAGE   = 50   # 2 hits = Death. 
POISON_DAMAGE = 10
MAX_STEPS     = 1000 # Increased to allow for more exploration

# -----------------------------------------------------------------------
# Maze Configuration
# -----------------------------------------------------------------------
# 25x25 Advanced Serpentine Maze
# S = Start (Top-Left), G = Goal (Bottom-Right), X = Pressure Plate
# T = 21-tile wide death corridor. Unsurvivable unless X is triggered.
MAZE_TEMPLATE = [
    "#################",
    "#S.#......X..PP.#",
    "#.##.############",
    "#.....#.........#",
    "###.#T#.###.#.###",
    "#...#T#.#...#...#",
    "#.###T#.#.###.#.#",
    "#.#...#...#...#P#",
    "#P#.#####.#.###.#",
    "#.#.#.....#.....#",
    "###.#.###.#.#.###",
    "#.....#P..#...#.#",
    "#.###.#.###.#...#",
    "#.P.#...#......G#",
    "#################",
]

PATROL_PATHS = [
    [(13, 5), (12, 5), (11, 5), (10, 5), (9, 5)],  # Enemy Bottom Left
    [(3,  10),  (3,  11),  (3,  12),  (3,  13),  (3,  14),  (3,  15)],  # Enemy Top Right
    [(7,  9),  (8,  9),  (9,  9),  (10,  9),  (11,  9)] # Enemy Bottom Right
]

# -----------------------------------------------------------------------
# Helper Classes
# -----------------------------------------------------------------------
class PatrolEnemy:
    def __init__(self, path):
        self.path = path
        self.idx = 0
        self.direction = 1

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

def parse_maze(template):
    char_map = {
        '#': WALL, '.': EMPTY, 'S': START, 'G': GOAL,
        'T': TRAP, 'P': POISON, 'X': PLATE, 'M': EMPTY, 'D': D_TRAP
    }
    return np.array([[char_map[c] for c in line] for line in template], dtype=np.int32)

# -----------------------------------------------------------------------
# Main Environment
# -----------------------------------------------------------------------
class DeathMazeEnv(_Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.initial_grid = parse_maze(MAZE_TEMPLATE)
        self.H, self.W = self.initial_grid.shape
        self.start_pos = tuple(zip(*np.where(self.initial_grid == START)))[0]
        self.goal_pos  = tuple(zip(*np.where(self.initial_grid == GOAL)))[0]
        self.enemies = [PatrolEnemy(path) for path in PATROL_PATHS]

        # Observation: includes a bit to tell the agent if the plate is on/off
        obs_size = 3 + 25 + 1 + 1
        self.observation_space = _Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)
        self.action_space = _Discrete(5)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = (int(self.start_pos[0]), int(self.start_pos[1]))
        self.health = MAX_HEALTH
        self.steps = 0
        self.done = False
        self.plate_activated = False
        self.current_grid = self.initial_grid.copy()
        self.visited = set()
        self.visited.add(self.agent_pos)
        for e in self.enemies: e.reset()
        return self._get_obs(), {}

    def _get_obs(self):
        r, c = self.agent_pos
        grid_view = self.current_grid.copy()
        for e in self.enemies:
            er, ec = e.pos
            grid_view[er, ec] = ENEMY

        patch = np.ones((5, 5), dtype=np.float32)
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.H and 0 <= nc < self.W:
                    patch[dr + 2, dc + 2] = grid_view[nr, nc] / 8.0

        dist = (abs(r - self.goal_pos[0]) + abs(c - self.goal_pos[1])) / (self.H + self.W)

        return np.array([
            r / self.H, c / self.W, self.health / MAX_HEALTH,
            *patch.flatten(), dist, float(self.plate_activated)
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = R_STEP
        info = {}
        prev_dist = abs(self.agent_pos[0]-self.goal_pos[0]) + abs(self.agent_pos[1]-self.goal_pos[1])

        dr, dc = ACTION_DELTAS[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        if action != WAIT:
            if not (0 <= nr < self.H and 0 <= nc < self.W) or self.current_grid[nr, nc] == WALL:
                reward += R_WALL
            else:
                self.agent_pos = (nr, nc)
                if self.agent_pos not in self.visited:
                    reward += R_EXPLORE
                    self.visited.add(self.agent_pos)

        new_dist = abs(self.agent_pos[0]-self.goal_pos[0]) + abs(self.agent_pos[1]-self.goal_pos[1])
        reward += R_CLOSER if new_dist < prev_dist else R_FARTHER

        tile = self.current_grid[self.agent_pos[0], self.agent_pos[1]]
        
        # PLATE LOGIC: No reward, just environment change
        if tile == PLATE and not self.plate_activated:
            self.plate_activated = True
            self.current_grid[self.current_grid == TRAP] = D_TRAP
            # print("Stepped on Plate")

        elif tile == TRAP:
            reward += R_TRAP
            self.health -= TRAP_DAMAGE
        #     if self.plate_activated:
        #         print("Trap deactivated by Plate!")
        # elif tile == D_TRAP:
        #     print("Stepped on Deactivated Trap")
            
        elif tile == POISON:
            reward += R_POISON
            self.health -= POISON_DAMAGE
            

        elif tile == GOAL:
            reward += R_GOAL
            self.done = True
            info["outcome"] = "goal"

        for e in self.enemies:
            e.step()
            if e.pos == self.agent_pos:
                reward += R_DEATH
                self.health = 0
                # print("Caught by Enemy!")

        if self.health <= 0:
            reward += R_DEATH
            self.done = True
            info["outcome"] = "dead"
        elif self.steps >= MAX_STEPS:
            self.done = True
            info["outcome"] = "timeout"

        return self._get_obs(), reward, self.done, (self.steps >= MAX_STEPS), info

    def render(self):
        char_map = {WALL: '#', EMPTY: '.', START: 'S', GOAL: 'G', 
                    TRAP: 'T', POISON: 'P', PLATE: 'X', D_TRAP: '_'}
        res = []
        for r in range(self.H):
            row = []
            for c in range(self.W):
                char = char_map.get(self.current_grid[r, c], '?')
                for e in self.enemies:
                    if (r, c) == e.pos: char = 'M'
                if (r, c) == self.agent_pos: char = 'A'
                row.append(char)
            res.append("".join(row))
        print(f"Step: {self.steps} | Health: {self.health} | Plate: {'ON' if self.plate_activated else 'OFF'}")
        print("\n".join(res) + "\n")