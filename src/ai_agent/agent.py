import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from src.ai_agent.agent_state_and_action import GameActions


# ----------------------------- MODEL: Deep Q-Network ----------------------------- #
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)  # Hidden layer
        self.out = nn.Linear(128, output_dim)  # Output layer (action space)

        # Weight Initialization (Xavier Initialization)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation for the first hidden layer
        x = torch.relu(self.fc2(x))  # ReLU activation for the second hidden layer
        return self.out(x)  # Final output (Q-values)


# ----------------------------- AGENT: Deep Q-Learning Agent ----------------------------- #
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.learning_rate = 0.0001

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=10000)
        self.batch_size = 256

        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        # reward = np.clip(reward, -1.0, 1.0)  # <- Add this line
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, episode):
        r0 = random.random()
        r1 = random.randint(0, 200)
        #  if r0 < self.epsilon or ((r1 % 10 == 0 or r1 % 5 == 0) and episode < 3000):
        if episode < 3000 or r0 < self.epsilon:
            # Exploration: Use normalized probabilities
            # 6 elements for all actions
            # No_action = 0
            # MoveLeft = 1
            # MoveRight = 2
            # Jump = 3
            # Shoot = 4
            # Grenade = 5
            weights = [0.2, 0.1, 0.5, 0.4, 0.4, 0.1]
            total = sum(weights)
            probabilities = [w / total for w in weights]
            possible_actions = list(GameActions)
            return "Exploration", [r0, r1], np.random.choice(possible_actions, p=probabilities)
        else:
            # Exploitation: Use Q-network
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            return "Exploitation", [r0, r1], torch.argmax(q_values).item()
            # q_values_np = q_values.detach().cpu().numpy().flatten()
            # temperature = max(0.5, 1.0 * (0.99 ** episode))  # anneal temperature
            # probs = self.softmax(q_values_np, temperature)  # You can tune the temperature
            # action_index = np.random.choice(len(probs), p=probs)
            # return "Exploitation", [r0, r1], action_index

    def softmax(self, x, temperature=1.0):
        """Optional: Softmax function if you ever want to convert Q-values to probabilities."""
        x = np.array(x)
        exp_x = np.exp(x / temperature)
        return exp_x / np.sum(exp_x)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Converting array to numpy array
        states = np.array(states)
        next_states = np.array(next_states)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Bellman Equation: Q(s,a) = r + gamma * max_a' Q_target(s', a')
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.SmoothL1Loss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0) #
        self.optimizer.step()

        # Decay epsilon after each replay to slowly transition from exploration to exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class RewardAI:
    def __init__(self):
        self.total_reward = 0

    def calculate_reward(self, prev_health, current_health, killed_enemy=False,
                        fired_bullet=False, bullet_hit_enemy=False, threw_grenade=False,
                        fell_or_hit_water=False, reached_exit=False, walked_forward=False):
        reward = 0.0

        # Reduced time penalty (encourage faster completion)
        reward -= 0.0001  # Reduced from 0.001

        # Penalty for dangerous falls/liquid collisions
        if fell_or_hit_water:
            reward -= 1.0  # Original penalty maintained

        # Health-based rewards
        health_change = current_health - prev_health
        if health_change > 0:
            reward += 0.1 * health_change  # Reward for gaining health
        elif health_change < 0:
            reward += 0.2 * health_change  # Penalize health loss more than we reward gains

        # Combat rewards
        if killed_enemy:
            reward += 5.0  # Substantial reward for eliminations
        if fired_bullet:
            reward += 0.3  # Reduced from 0.6 to discourage spamming
        if bullet_hit_enemy:
            reward += 2.0  # Reward accuracy
        if threw_grenade:
            reward += 0.5  # Reduced from 0.9 to prevent grenade spam

        # Movement incentives
        if walked_forward:
            reward += 0.5  # Increased from 0.1 to encourage progression

        # Strategic objective
        if reached_exit:
            reward += 50.0  # Massive reward for completing level (increased from 20.0)

        self.total_reward += reward
        return reward

    def reset_total_reward(self):
        self.total_reward = 0

    def calculate_total_reward(self):
        return self.total_reward