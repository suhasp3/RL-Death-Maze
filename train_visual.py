"""
Death Maze RL — Training with Pygame visualization.

Run:  python train_visual.py

A window opens immediately. Training runs in the background while the window
shows a "Training in Progress" screen. At each checkpoint episode the agent
pauses training and plays one full greedy episode so you can watch how it
improves from random thrashing to competent navigation.

Checkpoint episodes: 1, 50, 200, 500, 1000, 2000, 3000
"""

import os
import sys
import numpy as np
import pygame

from maze_env import DeathMazeEnv
from dqn_agent import DQNAgent
from visualizer import MazeVisualizer

# Episodes at which we pause and show a live demo
DEMO_EPISODES = {1, 100, 500, 1000, 2000, 4000, 6000, 8000}
N_EPISODES    = 8000
DEMO_FPS      = 8   # slow enough to watch
TRAIN_UPDATE  = 50  # refresh training screen every N episodes


def main():
    os.makedirs("outputs", exist_ok=True)

    # Two separate env instances: one for training, one for demos
    train_env = DeathMazeEnv()
    demo_env  = DeathMazeEnv()

    obs_size = train_env.observation_space.shape[0]
    agent = DQNAgent(
        obs_size=obs_size,
        n_actions=5,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.085,
        epsilon_decay=0.99995,
        batch_size=64,
        target_update_freq=500,
    )

    vis = MazeVisualizer(demo_env, fps=DEMO_FPS)

    print("=" * 55)
    print("  Death Maze RL  —  Visual Training")
    print("=" * 55)
    print(f"Checkpoints: {sorted(DEMO_EPISODES)}")
    print("Close window or press ESC to stop.\n")

    episode_rewards = []

    for ep in range(1, N_EPISODES + 1):

        # --- Demo checkpoint ---
        if ep in DEMO_EPISODES:
            print(f"[Demo] Episode {ep:4d}  epsilon={agent.epsilon:.3f}")
            outcome = vis.play_episode(agent, ep, fps=DEMO_FPS, training=True)
            print(f"         outcome: {outcome}")

        # --- One training episode ---
        obs, _ = train_env.reset()
        total_reward = 0.0
        while True:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, _ = train_env.step(action)
            agent.store(obs, action, reward, next_obs, float(terminated or truncated))
            agent.learn()
            obs = next_obs
            total_reward += reward
            if terminated or truncated:
                break
        episode_rewards.append(total_reward)

        # --- Keep window alive + show progress ---
        if ep % TRAIN_UPDATE == 0:
            avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            vis.show_training_screen(ep, N_EPISODES, avg, agent.epsilon)

            if ep % 500 == 0:
                print(f"  Episode {ep:5d} | avg(100): {avg:+8.1f} | epsilon: {agent.epsilon:.3f}")

    # --- Final demo ---
    print("\nTraining complete! Showing final demo...")
    vis.play_episode(agent, N_EPISODES, fps=6, training=False)

    agent.save("outputs/dqn_visual.pt")
    print("Model saved to outputs/dqn_visual.pt")
    vis.close()


if __name__ == "__main__":
    main()
