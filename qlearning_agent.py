"""
Q-Learning Baseline Agent for Death Maze RL
Tabular Q-learning with epsilon-greedy exploration.
"""

import numpy as np
import pickle
from collections import defaultdict


class QLearningAgent:
    """
    Tabular Q-learning agent.
    State is discretized as (agent_row, agent_col, health_bucket).
    """

    def __init__(
        self,
        n_actions=5,
        learning_rate=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-table: state -> array of Q-values per action
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def _discretize(self, obs):
        """
        Compress the observation into a hashable state key.
        Uses agent position + health bucket (0-4).
        """
        agent_r = int(obs[0] * 15)
        agent_c = int(obs[1] * 15)
        health_bucket = int(obs[2] * 4)  # 0-4
        return (agent_r, agent_c, health_bucket)

    def select_action(self, obs, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        state = self._discretize(obs)
        return int(np.argmax(self.q_table[state]))

    def update(self, obs, action, reward, next_obs, terminated):
        state      = self._discretize(obs)
        next_state = self._discretize(next_obs)

        current_q = self.q_table[state][action]
        if terminated:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action] += self.lr * (target - current_q)

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table saved to {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions), data)
        print(f"Q-table loaded from {path}")


def train_qlearning(env, agent, n_episodes=3000, print_every=500):
    episode_rewards = []
    episode_lengths = []
    outcomes = {"goal": 0, "dead": 0, "timeout": 0}

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, next_obs, terminated)
            obs = next_obs
            total_reward += reward
            steps += 1
            if terminated or truncated:
                outcome = info.get("outcome", "timeout")
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if ep % print_every == 0:
            recent = episode_rewards[-print_every:]
            print(
                f"Episode {ep:5d} | "
                f"Avg Reward: {np.mean(recent):8.1f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Goals: {outcomes['goal']} | "
                f"Deaths: {outcomes['dead']}"
            )
            outcomes = {"goal": 0, "dead": 0, "timeout": 0}

    return episode_rewards, episode_lengths
