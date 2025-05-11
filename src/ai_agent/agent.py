import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, output_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.position = 0
        self.priorities = [] #np.zeros((capacity,), dtype=np.float32)
        self.epsilon = 1e-5

    def add(self, error, sample):
        priority = (abs(error) + self.epsilon) ** self.alpha
        max_priority = max(self.priorities) if len(self.priorities) > 0 else priority
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.position] = sample
        if self.position < len(self.priorities):
            self.priorities[self.position] = max_priority
        else:
            self.priorities.append(max_priority)

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from an empty buffer")

        # Ensure priorities and buffer are synchronized
        if len(self.buffer) != len(self.priorities):
            print("[Error] Buffer and priorities length mismatch. Resetting priorities.")
            self.priorities = [1.0] * len(self.buffer)

        # Convert priorities to probabilities
        scaled_priorities = np.array(self.priorities, dtype=np.float32) ** self.alpha
        sum_priorities = np.sum(scaled_priorities)

        # Avoid NaN by resetting priorities if they collapse to zero
        if sum_priorities == 0 or np.isnan(sum_priorities):
            print("[Warning] Sum of priorities is zero or NaN. Resetting priorities.")
            self.priorities = [1.0] * len(self.buffer)
            scaled_priorities = np.array(self.priorities, dtype=np.float32) ** self.alpha
            sum_priorities = np.sum(scaled_priorities)

        # Normalize to get probabilities
        probabilities = scaled_priorities / sum_priorities

        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Calculate importance-sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** -beta
        weights /= weights.max()

        samples = [self.buffer[idx] for idx in indices]
        return samples, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + self.epsilon) ** self.alpha

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        # State and action dimensions (not hyperparameters, but part of the environment definition)
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Hyperparameters
        self.gamma = 0.99  # Discount factor for future rewards
        self.epsilon = 1.0  # Initial exploration rate (for epsilon-greedy policy)
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.997  # Rate at which epsilon decays after each episode
        self.learning_rate = 0.001  # Learning rate for the optimizer
        self.batch_size = 128  # Number of samples per training batch
        self.memory_size = 10000  # Maximum size of the replay buffer

        # Device configuration (choosing between GPU and CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Loss function (Mean Squared Error, used for TD error calculation in DQN)
        self.criterion = nn.MSELoss(reduction='none')

        # Replay buffer for prioritized experience replay
        self.memory = PrioritizedReplayBuffer(self.memory_size)

        # Q-network (main network for learning Q-values)
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)

        # Target network (used for more stable Q-value targets)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)

        # Optimizer for training the Q-network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Initial target network synchronization
        self.update_target_network()

    def act(self, state):
        if random.random() < self.epsilon:
        # if self.epsilon > 0.01:
            # high_priority_action = self.get_high_priority_action(state)
            # if high_priority_action is not None:
            #     return "Exploration (Memory)", None, high_priority_action

            # Fallback to pure random action
            action = random.randint(0, self.action_dim - 1)
            return "Exploration (Random)", None, action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).cpu().numpy().flatten()

            # Check if the Q-value for the best action is negative
            action = np.argmax(q_values)

            # print("Exploitation", action, q_values)

            # if q_values[action] < 0:
            #     # If the best action has a negative Q-value, explore instead
            #     action = random.randint(0, self.action_dim - 1)
            #     return "Exploration (Negative Q-value)", q_values, action

            action = np.argmax(q_values)
            return "Exploitation", q_values, action

    def get_high_priority_action(self, state, similarity_threshold=0.8):
        if not hasattr(self, 'memory') or len(self.memory) == 0:
            return None

        best_action = None
        max_priority = -float('inf')

        # Search for a similar state in the memory
        for i in range(len(self.memory)):
            sample_state, sample_action, _, _, _ = self.memory.buffer[i]

            # Calculate cosine similarity
            similarity = np.dot(state, sample_state) / (np.linalg.norm(state) * np.linalg.norm(sample_state))

            if similarity > max_priority:
                max_priority = similarity

            if similarity > similarity_threshold:
                state_tensor = torch.FloatTensor(sample_state).unsqueeze(0).to(self.device)

                # Check for NaNs in state_tensor
                if torch.any(torch.isnan(state_tensor)) or torch.any(torch.isinf(state_tensor)):
                    print(f"Warning: Sample state contains NaN or Inf values: {sample_state}")
                    continue

                # Get Q-values and check for NaNs
                q_values = self.q_network(state_tensor).detach().cpu().numpy().flatten()
                q_values = np.nan_to_num(q_values, nan=0.0)  # Replace NaNs with 0 or a small value
                max_q_value = q_values[sample_action]

                # print(f"------------Exploration with Memory] max_q_value: {max_q_value}, max_priority: {max_priority}")

                if max_q_value > max_priority:
                    max_priority = max_q_value
                    best_action = sample_action

        if best_action is not None:
            print("[Exploration with Memory] Using high-priority past action.")

        print()
        return best_action

    def remember(self, state, action, reward, next_state, done):
        with torch.no_grad():
            # Unpack the tuple
            state_dict, state_vector = next_state

            # Use the flattened state vector
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)

            q_values = self.q_network(state_tensor)
            next_q_values = self.target_network(next_state_tensor).detach().max(1)[0]
            target = reward + self.gamma * next_q_values * (1 - done)
            error = target - q_values[0, action]

        self.memory.add(error.item(), (state, action, reward, state_vector, done))

    def replay(self, beta=0.4):
        if len(self.memory) < self.batch_size:
            return
        samples, indices, weights = self.memory.sample(self.batch_size, beta)
        states, actions, rewards, next_states, dones = zip(*samples)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_network(next_states).detach().max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        errors = targets - q_values
        loss = (torch.FloatTensor(weights).to(self.device) * self.criterion(q_values, targets)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        errors = (targets - q_values).detach().cpu().numpy()

        # Replace NaNs or Infs with a small positive value
        errors = np.nan_to_num(errors, nan=1e-5, posinf=1e-5, neginf=1e-5)
        self.memory.update_priorities(indices, errors)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        if self.epsilon % 5 == 0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
