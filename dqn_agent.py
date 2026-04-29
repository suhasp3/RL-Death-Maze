"""
Deep Q-Network (DQN) Agent for Death Maze RL
Includes: replay buffer, target network, epsilon-greedy exploration.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


# -----------------------------------------------------------------------
# Neural Network
# -----------------------------------------------------------------------

class DQNNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------------
# Replay Buffer
# -----------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs,      dtype=np.float32),
            np.array(actions,  dtype=np.int64),
            np.array(rewards,  dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones,    dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# -----------------------------------------------------------------------
# DQN Agent
# -----------------------------------------------------------------------

class DQNAgent:
    def __init__(
        self,
        obs_size,
        n_actions=5,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        batch_size=64,
        target_update_freq=500,
        replay_capacity=50_000,
        min_replay_size=1000,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.min_replay_size = min_replay_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQNNetwork(obs_size, n_actions).to(self.device)
        self.target_net = DQNNetwork(obs_size, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(replay_capacity)
        self.steps_done = 0
        self.losses = []

    def select_action(self, obs, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.policy_net(obs_t)
        return int(q_vals.argmax(dim=1).item())

    def store(self, obs, action, reward, next_obs, done):
        self.replay.push(obs, action, reward, next_obs, done)

    def learn(self):
        if len(self.replay) < self.min_replay_size:
            return None

        obs, actions, rewards, next_obs, dones = self.replay.sample(self.batch_size)

        obs_t      = torch.FloatTensor(obs).to(self.device)
        actions_t  = torch.LongTensor(actions).to(self.device)
        rewards_t  = torch.FloatTensor(rewards).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t    = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        q_values = self.policy_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values (Double DQN style)
        with torch.no_grad():
            next_actions = self.policy_net(next_obs_t).argmax(dim=1)
            next_q = self.target_net(next_obs_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.steps_done += 1

        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"DQN model saved to {path}")

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"DQN model loaded from {path}")


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------

def train_dqn(env, agent, n_episodes=3000, print_every=500):
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
            done = terminated or truncated
            agent.store(obs, action, reward, next_obs, float(done))
            agent.learn()
            obs = next_obs
            total_reward += reward
            steps += 1
            if done:
                outcome = info.get("outcome", "timeout")
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if ep % print_every == 0:
            recent = episode_rewards[-print_every:]
            avg_loss = (
                np.mean(agent.losses[-1000:]) if agent.losses else 0.0
            )
            print(
                f"Episode {ep:5d} | "
                f"Avg Reward: {np.mean(recent):8.1f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Goals: {outcomes['goal']} | "
                f"Deaths: {outcomes['dead']}"
            )
            outcomes = {"goal": 0, "dead": 0, "timeout": 0}

    return episode_rewards, episode_lengths
