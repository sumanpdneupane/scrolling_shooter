import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from src.ai_agent.agent_state_and_action import GameActions

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)  # Second hidden layer
        self.out = nn.Linear(64, output_dim)  # Output layer

        # Weight Initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.learning_rate = 0.001
        self.update_target_every = 50
        self.batch_size = 128
        self.memory_size = 10000
        self.steps_per_episode = 0  # Track steps per episode

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=self.memory_size)
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.update_target_network()

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_size:
            self.memory.pop()
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state_dict, state, episode):
        self.steps_per_episode += 1  # Track steps

        if random.random() < self.epsilon:
            return self._explore(state_dict, episode)
        else:
            return self._exploit(state)

    def _explore(self, state_dict, episode):
        # Unified jump conditions
        jump_required = (
                state_dict["water_ahead"] or
                state_dict["water_below"] or
                state_dict["space_ahead"] or
                state_dict["on_edge"] or
                not state_dict["path_clear"]
        )
        if jump_required:
            # 60% chance to jump when obstacles/water present
            weights = [0.05, 0.05, 0.6, 0.1, 0.1, 0.1]  # [LEFT, RIGHT, JUMP, ...]
        elif state_dict["ground_distance"] > 0.4:
            weights = [0.1, 0.1, 0.6, 0.1, 0.1, 0.1]
        elif state_dict["path_clear"]:
            # Higher chance to move right when path is clear
            weights = [0.1, 0.5, 0.1, 0.1, 0.1, 0.1]
        else:
            weights = [0.2, 0.2, 0.2, 0.2, 0.2, 0.1]

        probabilities = np.array(weights) / sum(weights)
        return "Exploration", probabilities, np.random.choice(list(GameActions), p=probabilities)

    def _exploit(self, state):
        with torch.no_grad():
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)

            # Ensure proper shape
            if len(state.shape) == 1:
                if state.shape[0] > self.state_dim:
                    state = state[:self.state_dim]
                elif state.shape[0] < self.state_dim:
                    padded_state = np.zeros(self.state_dim, dtype=np.float32)
                    padded_state[:state.shape[0]] = state
                    state = padded_state

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).cpu().numpy().flatten()

        current_max_q = np.max(q_values)

        if np.isnan(current_max_q).any() or np.isinf(current_max_q).any():
            action_idx = np.random.randint(self.action_dim)
            return "Exploitation-Exploration", None, GameActions(action_idx)

        action_idx = np.argmax(q_values)
        print(f"[Exploit] Q-values: {q_values}, Action: {action_idx}")
        return "Exploitation", None, GameActions(action_idx)

    def replay(self, episode):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.tensor(np.array([m[0] for m in minibatch]), dtype=torch.float32).to(self.device)

        # Clip state if it's too long
        if states.shape[1] > self.state_dim:
            print(f"[Warning] Trimming state from {states.shape[1]} to {self.state_dim}")
            states = states[:, :self.state_dim]

        # Handle empty state
        if states.nelement() == 0:
            print("[Error] Empty states tensor. Skipping training step.")
            return

        actions = [m[1] for m in minibatch]

        # Convert GameActions enums to their integer value
        actions = [
            a.value if isinstance(a, GameActions) else int(a)
            # Convert GameActions to int, else cast numpy.int64 to int
            for a in actions
        ]

        actions = torch.tensor(actions, dtype=torch.long).to(self.device)

        rewards = torch.tensor([m[2] for m in minibatch], dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array([np.array(list(m[3][0].values())).flatten() for m in minibatch]), dtype=torch.float32).to(self.device)
        dones = torch.tensor([m[4] for m in minibatch], dtype=torch.float32).to(self.device)

        # Clip next state too if needed
        if next_states.shape[1] > self.state_dim:
            print(f"[Warning] Trimming next_state from {next_states.shape[1]} to {self.state_dim}")
            next_states = next_states[:, :self.state_dim]

        # Handle empty next_state
        if next_states.nelement() == 0:
            print("[Error] Empty next_states tensor. Skipping training step.")
            return

        # Q-learning target
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).detach().max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(q_values.squeeze(), targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Logging
        print(f"[Replay] Loss: {loss.item():.4f}, Epsilon: {self.epsilon:.4f}")

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def end_episode(self):
        """Call this when an episode ends"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps_per_episode = 0  # Reset step counter


# class DQNAgent:
#     def __init__(self, state_dim, action_dim):
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.gamma = 0.99
#         self.epsilon = 1.0  # Initial exploration probability
#         self.epsilon_min = 0.01  # Minimum exploration probability
#         self.epsilon_decay = 0.9995  # Slower decay rate
#         self.learning_rate = 0.0001
#         self.update_target_every = 100  # Target network update frequency
#         self.steps = 0
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.criterion = nn.MSELoss()
#
#         # Q-networks
#         self.q_network = QNetwork(state_dim, action_dim).to(self.device)
#         self.target_network = QNetwork(state_dim, action_dim).to(self.device)
#         self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
#         self.memory_size = 10000
#         self.memory = deque(maxlen=self.memory_size)  # Larger replay buffer
#         self.batch_size = 256  # Increased batch size
#
#         self.update_target_network()
#
#     def remember(self, state, action, reward, next_state, done):
#         # self.memory.append((state, action, reward, next_state, done))
#         """Stores experiences in the memory."""
#         if len(self.memory) >= self.memory_size:
#             self.memory.pop()  # Remove the oldest experience if memory is full
#         self.memory.append((state, action, reward, next_state, done))
#
#     def act(self, state_dict, state, episode):
#         # Dynamic exploration-exploitation balance
#         if random.random() < self.epsilon:
#             action_info = self._explore(state_dict, episode)
#         else:
#             action_info = self._exploit(state)
#         return action_info
#
#     def _explore(self, state_dict, episode):
#         # Dynamic weights based on the environment state
#         if state_dict["water_ahead"] or state_dict["water_below"] or state_dict["space_ahead"] or state_dict["on_edge"]:
#             # Prioritize jumping when water/hazard is detected
#             weights = [0.1, 0.2, 0.6, 0.1, 0.1, 0.1]
#         elif state_dict["ground_distance"] > 0.4:  # Moved this condition up
#             # Prioritize jumping for large gaps
#             weights = [0.1, 0.1, 0.6, 0.1, 0.1, 0.1]
#         elif not state_dict["path_clear"]:
#             # Handle obstacles/walls
#             weights = [0.1, 0.2, 0.6, 0.1, 0.1, 0.1]
#         elif state_dict["path_clear"]:
#             # Safe to move forward
#             weights = [0.1, 0.6, 0.1, 0.1, 0.1, 0.1]
#         else:
#             # Default exploration
#             weights = [0.2, 0.3, 0.2, 0.2, 0.2, 0.1]
#
#         probabilities = np.array(weights) / sum(weights)
#         return "Exploration", probabilities, np.random.choice(list(GameActions), p=probabilities)
#
#     def _exploit(self, state):
#         """Improved exploitation with low-reward randomness"""
#         with torch.no_grad():
#             # Existing state processing code
#             if not isinstance(state, np.ndarray):
#                 state = np.array(state, dtype=np.float32)
#
#             if len(state.shape) == 1 and state.shape[0] != self.state_dim:
#                 if state.shape[0] > self.state_dim:
#                     state = state[:self.state_dim]
#                 else:
#                     padded_state = np.zeros(self.state_dim, dtype=np.float32)
#                     padded_state[:state.shape[0]] = state
#                     state = padded_state
#
#             state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#             q_values = self.q_network(state_tensor).cpu().numpy().flatten()
#
#         # Store the maximum Q-value for this state
#         current_max_q = np.max(q_values)
#         # self.historic_q_values.append(current_max_q)
#
#         # Fallback 1: Invalid Q-values
#         if np.isnan(current_max_q).any() or np.isinf(current_max_q).any():
#             action_idx = np.random.randint(self.action_dim)
#             return "Exploitation-Exploration", None, GameActions(action_idx)
#
#         # Default behavior: argmax selection
#         action_idx = np.argmax(q_values)
#         return "Exploitation", None, GameActions(action_idx)
#
#     def replay(self, episode):
#         if len(self.memory) < self.batch_size:
#             return
#
#         self.steps += 1
#         batch = random.sample(self.memory, self.batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)
#
#         states_array = np.array([state[1] for state in states])
#         next_states_array = np.array([next_state[1] for next_state in next_states])
#
#         if states_array.dtype == np.dtype('O'):
#             states_list = []
#             for state in states:
#                 if isinstance(state, tuple) and len(state) > 1:
#                     state_value = state[1]
#                     if hasattr(state_value, '__iter__') and not isinstance(state_value, (str, bytes)):
#                         states_list.append(list(state_value))
#                     else:
#                         states_list.append([float(state_value)])
#                 else:
#                     try:
#                         states_list.append([float(state)])
#                     except:
#                         states_list.append([0.0])
#
#             states_array = np.array(states_list, dtype=np.float32)
#
#         if next_states_array.dtype == np.dtype('O'):
#             next_states_list = []
#             for next_state in next_states:
#                 if isinstance(next_state, tuple) and len(next_state) > 1:
#                     next_state_value = next_state[1]
#                     if hasattr(next_state_value, '__iter__') and not isinstance(next_state_value, (str, bytes)):
#                         next_states_list.append(list(next_state_value))
#                     else:
#                         next_states_list.append([float(next_state_value)])
#                 else:
#                     try:
#                         next_states_list.append([float(next_state)])
#                     except:
#                         next_states_list.append([0.0])
#
#             next_states_array = np.array(next_states_list, dtype=np.float32)
#
#         if len(states_array.shape) == 1:
#             states_array = states_array.reshape(-1, 1)
#             if self.state_dim > 1:
#                 states_array = np.repeat(states_array, self.state_dim, axis=1)
#
#         if len(next_states_array.shape) == 1:
#             next_states_array = next_states_array.reshape(-1, 1)
#             if self.state_dim > 1:
#                 next_states_array = np.repeat(next_states_array, self.state_dim, axis=1)
#
#         if states_array.shape[1] != self.state_dim:
#             states_array = np.resize(states_array, (self.batch_size, self.state_dim))
#
#         if next_states_array.shape[1] != self.state_dim:
#             next_states_array = np.resize(next_states_array, (self.batch_size, self.state_dim))
#
#         # Convert enum actions to integers if needed
#         if isinstance(actions[0], GameActions):
#             actions = [a.value for a in actions]
#
#         states = torch.FloatTensor(states_array.astype(np.float32)).to(self.device)
#         actions = torch.LongTensor(actions).to(self.device)
#         rewards = torch.FloatTensor(rewards).to(self.device)
#         next_states = torch.FloatTensor(next_states_array.astype(np.float32)).to(self.device)
#         dones = torch.FloatTensor(dones).to(self.device)
#
        # # Compute Q-values for current states
        # q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
#
#         # Compute Q-values for next states (target)
#         next_q_values = self.target_network(next_states).max(1)[0].detach()
#         target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
#
#         # Compute loss
#         loss = self.criterion(q_values.squeeze(), target_q_values)
#
#         # Backpropagation
#         self.optimizer.zero_grad()
#         loss.backward()
#
#         # Clip gradients for stability
#         torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
#         self.optimizer.step()
#
#         # Update target network
#         if self.steps % self.update_target_every == 0:
#             self.update_target_network()
#
#         # Decay epsilon
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay
#
#
#     def update_target_network(self):
#         """Copy weights from q_network to target_network"""
#         self.target_network.load_state_dict(self.q_network.state_dict())