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

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        self.epsilon = 1.0  # Initial exploration probability
        self.epsilon_min = 0.01  # Minimum exploration probability
        self.epsilon_decay = 0.9995  # Slower decay rate
        self.learning_rate = 0.0001
        self.update_target_every = 100  # Target network update frequency
        self.steps = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

        # New: Track rewards and Q-values for state-action pairs
        # self.state_action_reward_history = {}  # Format: {(state_hash, action): [rewards]}
        # self.historic_q_values = deque(maxlen=1000)
        # self.historic_rewards = deque(maxlen=1000)  # Track raw rewards

        # Q-networks
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.memory_size = 100000
        self.memory = deque(maxlen=self.memory_size)  # Larger replay buffer
        self.batch_size = 1024  # Increased batch size

        self.update_target_network()

    def remember(self, state, action, reward, next_state, done):
        # self.memory.append((state, action, reward, next_state, done))
        """Stores experiences in the memory."""
        if len(self.memory) >= self.memory_size:
            self.memory.pop()  # Remove the oldest experience if memory is full
        self.memory.append((state, action, reward, next_state, done))

        # # Track rewards for the current state-action pair
        # state_hash = self._hash_state(state)  # Convert state to a hashable key
        # key = (state_hash, action)
        # if key not in self.state_action_reward_history:
        #     self.state_action_reward_history[key] = []
        # self.state_action_reward_history[key].append(reward)
        # # Track global rewards for threshold calculations
        # self.historic_rewards.append(reward)

    def act(self, state_dict, state, episode):
        # Dynamic exploration-exploitation balance
        if random.random() < self.epsilon:  # Exploration
            return self._explore(state_dict, episode)
        else:  # Exploitation
            return self._exploit(state)

    def _explore(self, state_dict, episode):
        # Dynamic weights based on the environment state
        if state_dict["water_ahead"] or state_dict["water_below"] or state_dict["space_ahead"] or state_dict["tile_left_hit"] or state_dict["tile_right_hit"]:
            # Prioritize jumping when water is ahead
            weights = [0.1, 0.2, 0.6, 0.1, 0.1, 0.1]  # Encourage jumping when water is ahead
        elif not state_dict["path_clear"]:
            # Prioritize jumping when there is a wall ahead (path is not clear)
            weights = [0.1, 0.2, 0.6, 0.1, 0.1, 0.1]  # Encourage jumping over the obstacle
        elif state_dict["path_clear"]:
            # Prioritize forward movement when the path is clear and on the ground
            weights = [0.1, 0.6, 0.1, 0.1, 0.1, 0.1]  # Encourage moving forward
        elif state_dict["ground_distance"] > 0.4:
            # Prioritize jumping if there is a large gap
            weights = [0.1, 0.1, 0.6, 0.1, 0.1, 0.1]  # Jump over large gaps
        else:
            # Default exploration
            weights = [0.2 , 0.3, 0.2, 0.2, 0.2, 0.1]  # Balanced exploration

        # Normalize the weights and return a random action based on probabilities
        probabilities = np.array(weights) / sum(weights)
        return "Exploration", probabilities, np.random.choice(list(GameActions), p=probabilities)

    def _exploit(self, state):
        """Improved exploitation with low-reward randomness"""
        with torch.no_grad():
            # Existing state processing code
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)

            if len(state.shape) == 1 and state.shape[0] != self.state_dim:
                if state.shape[0] > self.state_dim:
                    state = state[:self.state_dim]
                else:
                    padded_state = np.zeros(self.state_dim, dtype=np.float32)
                    padded_state[:state.shape[0]] = state
                    state = padded_state

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).cpu().numpy().flatten()

        # Store the maximum Q-value for this state
        current_max_q = np.max(q_values)
        # self.historic_q_values.append(current_max_q)

        # Fallback 1: Invalid Q-values
        if np.isnan(current_max_q).any() or np.isinf(current_max_q).any():
            action_idx = np.random.randint(self.action_dim)
            return "Exploitation-Exploration", None, GameActions(action_idx)

        # Fallback 2: Check historical rewards for this state-action pair
        # action_idx = np.argmax(q_values)
        # selected_action = GameActions(action_idx)
        # rewards = self.get_rewards_for_state_action(state, selected_action)
        # if len(rewards) > 150:  # Require minimum samples
        #     avg_reward = np.mean(rewards)
        #     reward_threshold = np.percentile(self.historic_rewards, 100)  # 25th percentile
        #     if avg_reward < reward_threshold:
        #         return self._fallback_action(f"Low reward history: {avg_reward:.2f} < {reward_threshold:.2f}")
        #     print("--------------- rewards", avg_reward, reward_threshold, rewards)

        # Default behavior: argmax selection
        action_idx = np.argmax(q_values)
        return "Exploitation", None, GameActions(action_idx)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        self.steps += 1

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_array = np.array([state[1] for state in states])
        next_states_array = np.array([next_state[1] for next_state in next_states])

        if states_array.dtype == np.dtype('O'):
            states_list = []
            for state in states:
                if isinstance(state, tuple) and len(state) > 1:
                    state_value = state[1]
                    if hasattr(state_value, '__iter__') and not isinstance(state_value, (str, bytes)):
                        states_list.append(list(state_value))
                    else:
                        states_list.append([float(state_value)])
                else:
                    try:
                        states_list.append([float(state)])
                    except:
                        states_list.append([0.0])

            states_array = np.array(states_list, dtype=np.float32)

        if next_states_array.dtype == np.dtype('O'):
            next_states_list = []
            for next_state in next_states:
                if isinstance(next_state, tuple) and len(next_state) > 1:
                    next_state_value = next_state[1]
                    if hasattr(next_state_value, '__iter__') and not isinstance(next_state_value, (str, bytes)):
                        next_states_list.append(list(next_state_value))
                    else:
                        next_states_list.append([float(next_state_value)])
                else:
                    try:
                        next_states_list.append([float(next_state)])
                    except:
                        next_states_list.append([0.0])

            next_states_array = np.array(next_states_list, dtype=np.float32)

        if len(states_array.shape) == 1:
            states_array = states_array.reshape(-1, 1)
            if self.state_dim > 1:
                states_array = np.repeat(states_array, self.state_dim, axis=1)

        if len(next_states_array.shape) == 1:
            next_states_array = next_states_array.reshape(-1, 1)
            if self.state_dim > 1:
                next_states_array = np.repeat(next_states_array, self.state_dim, axis=1)

        if states_array.shape[1] != self.state_dim:
            states_array = np.resize(states_array, (self.batch_size, self.state_dim))

        if next_states_array.shape[1] != self.state_dim:
            next_states_array = np.resize(next_states_array, (self.batch_size, self.state_dim))

        states = torch.FloatTensor(states_array.astype(np.float32)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states_array.astype(np.float32)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q-values for current states
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Compute Q-values for next states (target)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = self.criterion(q_values.squeeze(), target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        if self.steps % self.update_target_every == 0:
            self.update_target_network()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """Copy weights from q_network to target_network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    # def _hash_state(self, state):
    #     """Convert state array/tensor to a hashable key (simplified example)."""
    #     return tuple(np.round(state, 2).flatten().tolist())  # Round to 2 decimals for grouping
    #
    # def get_rewards_for_state_action(self, state, action):
    #     """Return list of rewards received for this (state, action) pair."""
    #     state_hash = self._hash_state(state)
    #     key = (state_hash, action)
    #     reward = self.state_action_reward_history.get(key, [])
    #     # print("get_rewards_for_state_action------->", state_hash, reward)
    #     return reward
    #
    # def _fallback_action(self, reason):
    #     """Take random action with logging."""
    #     print(f"Fallback triggered: {reason}")
    #     action_idx = np.random.randint(self.action_dim)
    #     return "Exploitation-Fallback", None, GameActions(action_idx)