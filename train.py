"""
Death Maze RL — Main Training Script
Trains both Q-Learning and DQN agents, then plots comparison curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

from maze_env import DeathMazeEnv
from qlearning_agent import QLearningAgent, train_qlearning
from dqn_agent import DQNAgent, train_dqn


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def smooth(rewards, window=50):
    """Rolling mean for cleaner learning curves."""
    smoothed = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(rewards[start:i+1]))
    return smoothed


def evaluate_agent(env, agent, n_episodes=100, is_dqn=False):
    """Run agent greedily and return avg reward + win rate."""
    rewards, wins = [], 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0
        while True:
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                if info.get("outcome") == "goal":
                    wins += 1
                break
        rewards.append(total)
    return np.mean(rewards), wins / n_episodes


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def plot_comparison(ql_rewards, dqn_rewards, save_path="learning_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Death Maze RL — Q-Learning vs DQN", fontsize=14, fontweight='bold')

    for ax, rewards, label, color in zip(
        axes,
        [ql_rewards, dqn_rewards],
        ["Q-Learning", "DQN"],
        ["steelblue", "darkorange"],
    ):
        episodes = list(range(1, len(rewards) + 1))
        ax.plot(episodes, rewards, alpha=0.25, color=color, linewidth=0.8)
        ax.plot(episodes, smooth(rewards, 50), color=color, linewidth=2, label="Smoothed (50 ep)")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved learning curves → {save_path}")
    plt.close()


def plot_combined(ql_rewards, dqn_rewards, save_path="combined_curves.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    episodes_ql  = list(range(1, len(ql_rewards)  + 1))
    episodes_dqn = list(range(1, len(dqn_rewards) + 1))

    ax.plot(episodes_ql,  smooth(ql_rewards,  50), color="steelblue",   linewidth=2, label="Q-Learning")
    ax.plot(episodes_dqn, smooth(dqn_rewards, 50), color="darkorange",  linewidth=2, label="DQN")
    ax.set_title("Death Maze RL — Algorithm Comparison (Smoothed)", fontsize=13)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward (smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved combined curves → {save_path}")
    plt.close()


def plot_maze_layout(save_path="maze_layout.png"):
    """Visualize the static maze layout."""
    from maze_env import MAZE_TEMPLATE, WALL, START, GOAL, TRAP, POISON, EMPTY

    color_map = {
        '#': '#2c2c2c',  # wall
        '.': '#f5f5f5',  # empty
        'S': '#4caf50',  # start
        'G': '#ffd700',  # goal
        'T': '#f44336',  # trap
        'P': '#9c27b0',  # poison
        'M': '#2196f3',  # enemy
    }
    label_map = {
        '#': 'Wall', '.': 'Empty', 'S': 'Start', 'G': 'Goal',
        'T': 'Trap', 'P': 'Poison', 'M': 'Enemy',
    }

    rows = len(MAZE_TEMPLATE)
    cols = len(MAZE_TEMPLATE[0])
    img = np.zeros((rows, cols, 3))

    for r, line in enumerate(MAZE_TEMPLATE):
        for c, ch in enumerate(line):
            hex_color = color_map.get(ch, '#ffffff')
            rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) / 255 for i in (0, 2, 4))
            img[r, c] = rgb

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, interpolation='nearest')
    ax.set_title("Death Maze — Level Layout", fontsize=13, fontweight='bold')
    ax.axis('off')

    # Legend
    patches = [mpatches.Patch(color=color_map[k], label=label_map[k])
               for k in ['S', 'G', '#', '.', 'T', 'P', 'M']]
    ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.25, 1), fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved maze layout → {save_path}")
    plt.close()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("  Death Maze RL")
    print("=" * 60)

    # --- Maze layout ---
    plot_maze_layout("outputs/maze_layout.png")

    # ---------------------------------------------------------------
    # 1. Q-Learning
    # ---------------------------------------------------------------
    print("\n[1/2] Training Q-Learning agent (3000 episodes)...")
    env_ql = DeathMazeEnv()
    ql_agent = QLearningAgent(
        n_actions=5,
        learning_rate=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
    )
    ql_rewards, ql_lengths = train_qlearning(env_ql, ql_agent, n_episodes=3000, print_every=500)
    ql_agent.save("outputs/qtable.pkl")

    ql_eval_reward, ql_win_rate = evaluate_agent(env_ql, ql_agent, n_episodes=100)
    print(f"Q-Learning Eval → Avg Reward: {ql_eval_reward:.1f} | Win Rate: {ql_win_rate*100:.1f}%")

    # ---------------------------------------------------------------
    # 2. DQN
    # ---------------------------------------------------------------
    print("\n[2/2] Training DQN agent (3000 episodes)...")
    env_dqn = DeathMazeEnv()
    obs_size = env_dqn.observation_space.shape[0]
    dqn_agent = DQNAgent(
        obs_size=obs_size,
        n_actions=5,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        batch_size=64,
        target_update_freq=500,
    )
    dqn_rewards, dqn_lengths = train_dqn(env_dqn, dqn_agent, n_episodes=3000, print_every=500)
    dqn_agent.save("outputs/dqn_model.pt")

    dqn_eval_reward, dqn_win_rate = evaluate_agent(env_dqn, dqn_agent, n_episodes=100, is_dqn=True)
    print(f"DQN Eval → Avg Reward: {dqn_eval_reward:.1f} | Win Rate: {dqn_win_rate*100:.1f}%")

    # ---------------------------------------------------------------
    # 3. Plots
    # ---------------------------------------------------------------
    plot_comparison(ql_rewards, dqn_rewards, "outputs/learning_curves.png")
    plot_combined(ql_rewards, dqn_rewards,   "outputs/combined_curves.png")

    # ---------------------------------------------------------------
    # 4. Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':<15} {'Avg Reward':>12} {'Win Rate':>10}")
    print("-" * 40)
    print(f"{'Q-Learning':<15} {ql_eval_reward:>12.1f} {ql_win_rate*100:>9.1f}%")
    print(f"{'DQN':<15} {dqn_eval_reward:>12.1f} {dqn_win_rate*100:>9.1f}%")
    print("=" * 60)
    print("\nAll outputs saved to ./outputs/")


if __name__ == "__main__":
    main()
