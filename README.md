# Death Maze RL 🏴‍☠️

**DATA 522 Final Project — Reinforcement Learning on a Dynamic Maze Environment**

> Train RL agents (Q-Learning & DQN) to navigate a hazardous 15×15 maze filled with traps, poison tiles, and patrolling enemies.
<img width="478" height="452" alt="image" src="https://github.com/user-attachments/assets/e8a6409b-96b4-4a24-b3fb-6e1036adddc3" />

---

## Why RL (Not Just Pathfinding)?

A static maze can be solved with BFS/A*. Death Maze requires RL because:
- **Patrolling enemies** create timing decisions — the agent must wait or route around patrol cycles
- **Health management** introduces long-term planning (avoid poison even if it's the short path)
- **Trap tiles** require learning from penalty feedback
- **Sequential decision making** across hundreds of steps with delayed rewards

---

## Environment

### Tile Legend
| Tile | Symbol | Effect |
|------|--------|--------|
| Wall | `#` | Impassable |
| Empty | `.` | Free to walk |
| Start | `S` | Episode start |
| Goal | `G` | +100 reward, episode ends |
| Trap | `T` | -20 reward, -30 HP |
| Poison | `P` | -5 reward, -10 HP |
| Enemy | `M` | Instant death if touched |
| Agent | `A` | Your agent |

### Reward Function
| Event | Reward |
|-------|--------|
| Reach goal | +100 |
| Each step | -1 |
| Hit wall | -3 |
| Hit trap | -20 |
| Hit poison | -5 |
| Death (HP ≤ 0 or enemy) | -100 |
| Move closer to goal | +0.5 |
| Move farther from goal | -0.5 |

### Observation Space (29-dim vector)
- Agent row, col (normalized)
- Health (normalized)
- 5×5 local patch around agent (25 values, normalized tile IDs)
- Manhattan distance to goal (normalized)

### Action Space
`Discrete(5)` — UP, DOWN, LEFT, RIGHT, WAIT

---

## Algorithms

### 1. Q-Learning (Baseline)
- Tabular Q-learning with ε-greedy exploration
- State discretized as `(row, col, health_bucket)`
- ε decays from 1.0 → 0.05 over training

### 2. DQN (Advanced)
- 3-layer MLP: 128 → 128 → 64 → n_actions
- Experience replay buffer (50k capacity)
- Target network (synced every 500 steps)
- Double DQN-style updates
- Gradient clipping (norm ≤ 10)

---

## Project Structure

```
death_maze_rl/
├── maze_env.py          # Gym environment (MDP, tiles, patrol enemies)
├── qlearning_agent.py   # Tabular Q-learning agent + training loop
├── dqn_agent.py         # DQN agent (network, replay buffer, training loop)
├── train.py             # Main script: trains both agents, plots results
└── README.md
```

---

## Installation

```bash
pip install gymnasium numpy torch matplotlib
```

---

## Usage

```bash
cd death_maze_rl
python train.py
```

Outputs saved to `./outputs/`:
- `maze_layout.png` — visual map of the maze
- `learning_curves.png` — side-by-side Q-Learning vs DQN curves
- `combined_curves.png` — overlaid comparison
- `qtable.pkl` — saved Q-table
- `dqn_model.pt` — saved DQN weights

---

## Connection to Course Material (DATA 522 — Lec 15)

| Concept | Implementation |
|---------|---------------|
| MDP (State, Action, Reward, Transition) | `DeathMazeEnv` — full MDP formulation |
| Policy π(a\|s) | ε-greedy policy in both agents |
| Bellman Equation | Q-update rule in `QLearningAgent.update()` |
| Value function / Q-function | `q_table` (tabular) and `DQNNetwork` (neural) |
| Reward shaping | Step penalty + distance shaping |
| Reward hacking prevention | Balanced reward magnitudes |
| Exploration-exploitation | ε-decay schedule |
| Replay buffer | `ReplayBuffer` in DQN |
| Target network | Stable Bellman targets |


# RL-Death-Maze
