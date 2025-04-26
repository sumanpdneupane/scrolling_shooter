import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# ----------------------------- MODEL: Deep Q-Network ----------------------------- #
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128) # Hidden layer
        self.fc3 = nn.Linear(128, 128)  # Hidden layer
        self.fc4 = nn.Linear(128, 128)  # Hidden layer
        self.fc5 = nn.Linear(128, 128)  # Hidden layer
        self.out = nn.Linear(128, output_dim)  # Output layer (action space)

        # Weight Initialization (Xavier Initialization)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation for the first hidden layer
        x = torch.relu(self.fc2(x))  # ReLU activation for the second hidden layer
        x = torch.relu(self.fc3(x))  # ReLU activation for the second hidden layer
        x = torch.relu(self.fc4(x))  # ReLU activation for the second hidden layer
        x = torch.relu(self.fc5(x))  # ReLU activation for the second hidden layer
        return self.out(x)  # Final output (Q-values)


# ----------------------------- AGENT: Deep Q-Learning Agent ----------------------------- #
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        # self.learning_rate = 0.0005

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.update_target_network()


    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        rand = random.random()
        if rand < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def replay(self, episode):
        self.episode = episode
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Converting array to numpy array
        states = np.array(states)
        next_states = np.array(next_states)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Bellman Equation: Q(s,a) = r + gamma * max_a' Q_target(s', a')
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        # loss = nn.MSELoss()(q_values, targets)
        loss = nn.SmoothL1Loss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class RewardAI:
    def __init__(self):
        self.total_reward = 0

    def calculate_reward(self, prev_health, current_health, killed_enemy=False, fired_bullet=False, bullet_hit_enemy= False, threw_grenade= False, fell_or_hit_water= False, reached_exit= False, walked_forward= False):
        reward = 0.0

        # Always give time penalty
        reward -= 0.001

        # Penalty for falling into water or off the ground
        if fell_or_hit_water:
            reward -= 1.0

        # Reward/Punish health change
        if current_health > prev_health:
            reward += 0.1
        elif current_health < prev_health:
            reward -= 0.1

        # Bonus rewards
        if killed_enemy:
            reward += 2.0
        if fired_bullet:
            reward += 0.6
        if bullet_hit_enemy:
            reward += 1.0
        if threw_grenade:
            reward += 0.9
        if walked_forward:
            reward += 0.05

        # HIGH reward for reaching the exit
        if reached_exit:
            reward += 5.0

        self.total_reward += reward
        return reward

    def reset_total_reward(self):
        self.total_reward = 0

    def calculate_total_reward(self):
        return self.total_reward
