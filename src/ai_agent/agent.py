# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import random
#
# class QNetwork(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(QNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.out = nn.Linear(128, output_dim)
#
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.xavier_uniform_(self.out.weight)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.out(x)
#
# class PrioritizedReplayBuffer:
#     def __init__(self, capacity, alpha=0.6):
#         self.capacity = capacity
#         self.alpha = alpha
#         self.buffer = []
#         self.position = 0
#         self.priorities = [] #np.zeros((capacity,), dtype=np.float32)
#         self.epsilon = 1e-5
#
#     def add(self, error, sample):
#         priority = (abs(error) + self.epsilon) ** self.alpha
#         max_priority = max(self.priorities) if len(self.priorities) > 0 else priority
#         if len(self.buffer) < self.capacity:
#             self.buffer.append(sample)
#         else:
#             self.buffer[self.position] = sample
#         if self.position < len(self.priorities):
#             self.priorities[self.position] = max_priority
#         else:
#             self.priorities.append(max_priority)
#
#         self.position = (self.position + 1) % self.capacity
#
#     def sample(self, batch_size, beta=0.4):
#         if len(self.buffer) == 0:
#             raise ValueError("Cannot sample from an empty buffer")
#
#         # Ensure priorities and buffer are synchronized
#         if len(self.buffer) != len(self.priorities):
#             print("[Error] Buffer and priorities length mismatch. Resetting priorities.")
#             self.priorities = [1.0] * len(self.buffer)
#
#         # Convert priorities to probabilities
#         scaled_priorities = np.array(self.priorities, dtype=np.float32) ** self.alpha
#         sum_priorities = np.sum(scaled_priorities)
#
#         # Avoid NaN by resetting priorities if they collapse to zero
#         if sum_priorities == 0 or np.isnan(sum_priorities):
#             print("[Warning] Sum of priorities is zero or NaN. Resetting priorities.")
#             self.priorities = [1.0] * len(self.buffer)
#             scaled_priorities = np.array(self.priorities, dtype=np.float32) ** self.alpha
#             sum_priorities = np.sum(scaled_priorities)
#
#         # Normalize to get probabilities
#         probabilities = scaled_priorities / sum_priorities
#
#         # Sample indices based on priorities
#         indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
#
#         # Calculate importance-sampling weights
#         weights = (len(self.buffer) * probabilities[indices]) ** -beta
#         weights /= weights.max()
#
#         samples = [self.buffer[idx] for idx in indices]
#         return samples, indices, weights
#
#     def update_priorities(self, indices, errors):
#         for idx, error in zip(indices, errors):
#             self.priorities[idx] = (abs(error) + self.epsilon) ** self.alpha
#
#     def __len__(self):
#         return len(self.buffer)
#
# class DQNAgent:
#     def __init__(self, state_dim, action_dim):
#         # State and action dimensions (not hyperparameters, but part of the environment definition)
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#
#         # Hyperparameters
#         self.gamma = 0.99  # Discount factor for future rewards
#         self.epsilon = 1.0  # Initial exploration rate (for epsilon-greedy policy)
#         self.epsilon_min = 0.01  # Minimum exploration rate
#         self.epsilon_decay = 0.997  # Rate at which epsilon decays after each episode
#         self.learning_rate = 0.001  # Learning rate for the optimizer
#         self.batch_size = 128  # Number of samples per training batch
#         self.memory_size = 10000  # Maximum size of the replay buffer
#
#         # Device configuration (choosing between GPU and CPU)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         # Loss function (Mean Squared Error, used for TD error calculation in DQN)
#         self.criterion = nn.MSELoss(reduction='none')
#
#         # Replay buffer for prioritized experience replay
#         self.memory = PrioritizedReplayBuffer(self.memory_size)
#
#         # Q-network (main network for learning Q-values)
#         self.q_network = QNetwork(state_dim, action_dim).to(self.device)
#
#         # Target network (used for more stable Q-value targets)
#         self.target_network = QNetwork(state_dim, action_dim).to(self.device)
#
#         # Optimizer for training the Q-network
#         self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
#
#         # Initial target network synchronization
#         self.update_target_network()
#
#     def act(self, state):
#         if random.random() < self.epsilon:
#         # if self.epsilon > 0.01:
#             # high_priority_action = self.get_high_priority_action(state)
#             # if high_priority_action is not None:
#             #     return "Exploration (Memory)", None, high_priority_action
#
#             # Fallback to pure random action
#             action = random.randint(0, self.action_dim - 1)
#             return "Exploration (Random)", None, action
#         with torch.no_grad():
#             state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#             q_values = self.q_network(state_tensor).cpu().numpy().flatten()
#
#             # Check if the Q-value for the best action is negative
#             action = np.argmax(q_values)
#
#             # print("Exploitation", action, q_values)
#
#             # if q_values[action] < 0:
#             #     # If the best action has a negative Q-value, explore instead
#             #     action = random.randint(0, self.action_dim - 1)
#             #     return "Exploration (Negative Q-value)", q_values, action
#
#             action = np.argmax(q_values)
#             return "Exploitation", q_values, action
#
#     def get_high_priority_action(self, state, similarity_threshold=0.8):
#         if not hasattr(self, 'memory') or len(self.memory) == 0:
#             return None
#
#         best_action = None
#         max_priority = -float('inf')
#
#         # Search for a similar state in the memory
#         for i in range(len(self.memory)):
#             sample_state, sample_action, _, _, _ = self.memory.buffer[i]
#
#             # Calculate cosine similarity
#             similarity = np.dot(state, sample_state) / (np.linalg.norm(state) * np.linalg.norm(sample_state))
#
#             if similarity > max_priority:
#                 max_priority = similarity
#
#             if similarity > similarity_threshold:
#                 state_tensor = torch.FloatTensor(sample_state).unsqueeze(0).to(self.device)
#
#                 # Check for NaNs in state_tensor
#                 if torch.any(torch.isnan(state_tensor)) or torch.any(torch.isinf(state_tensor)):
#                     print(f"Warning: Sample state contains NaN or Inf values: {sample_state}")
#                     continue
#
#                 # Get Q-values and check for NaNs
#                 q_values = self.q_network(state_tensor).detach().cpu().numpy().flatten()
#                 q_values = np.nan_to_num(q_values, nan=0.0)  # Replace NaNs with 0 or a small value
#                 max_q_value = q_values[sample_action]
#
#                 # print(f"------------Exploration with Memory] max_q_value: {max_q_value}, max_priority: {max_priority}")
#
#                 if max_q_value > max_priority:
#                     max_priority = max_q_value
#                     best_action = sample_action
#
#         if best_action is not None:
#             print("[Exploration with Memory] Using high-priority past action.")
#
#         print()
#         return best_action
#
#     def remember(self, state, action, reward, next_state, done):
#         with torch.no_grad():
#             # Unpack the tuple
#             state_dict, state_vector = next_state
#
#             # Use the flattened state vector
#             state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#             next_state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
#
#             q_values = self.q_network(state_tensor)
#             next_q_values = self.target_network(next_state_tensor).detach().max(1)[0]
#             target = reward + self.gamma * next_q_values * (1 - done)
#             error = target - q_values[0, action]
#
#         self.memory.add(error.item(), (state, action, reward, state_vector, done))
#
#     def replay(self, beta=0.4):
#         if len(self.memory) < self.batch_size:
#             return
#         samples, indices, weights = self.memory.sample(self.batch_size, beta)
#         states, actions, rewards, next_states, dones = zip(*samples)
#         states = torch.FloatTensor(states).to(self.device)
#         actions = torch.LongTensor(actions).to(self.device)
#         rewards = torch.FloatTensor(rewards).to(self.device)
#         next_states = torch.FloatTensor(next_states).to(self.device)
#         dones = torch.FloatTensor(dones).to(self.device)
#
#         q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
#         next_q_values = self.target_network(next_states).detach().max(1)[0]
#         targets = rewards + self.gamma * next_q_values * (1 - dones)
#
#         errors = targets - q_values
#         loss = (torch.FloatTensor(weights).to(self.device) * self.criterion(q_values, targets)).mean()
#
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#         errors = (targets - q_values).detach().cpu().numpy()
#
#         # Replace NaNs or Infs with a small positive value
#         errors = np.nan_to_num(errors, nan=1e-5, posinf=1e-5, neginf=1e-5)
#         self.memory.update_priorities(indices, errors)
#
#     def update_target_network(self):
#         self.target_network.load_state_dict(self.q_network.state_dict())
#
#     def decay_epsilon(self):
#         if self.epsilon % 5 == 0:
#             self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# import random
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, Flatten, Dense
# from tensorflow.keras.optimizers import Adam
#
# class PirorityBuffer:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.tree = np.zeros(2 * capacity - 1)
#         self.data = np.zeros(capacity, dtype=object)
#         self.size = 0
#         self.ptr = 0
#
#     def add(self, priority, data):
#         index = self.ptr + self.capacity - 1
#         self.data[self.ptr] = data
#         self.update(index, priority)
#         self.ptr = (self.ptr + 1) % self.capacity
#         self.size = min(self.size + 1, self.capacity)
#
#     def update(self, index, priority):
#         change = priority - self.tree[index]
#         self.tree[index] = priority
#         self._propagate(index, change)
#
#     def _propagate(self, index, change):
#         parent = (index - 1) // 2
#         self.tree[parent] += change
#         if parent != 0:
#             self._propagate(parent, change)
#
#     def get(self, value):
#         index = self._retrieve(0, value)
#         data_index = index - self.capacity + 1
#         return index, self.tree[index], self.data[data_index]
#
#     def _retrieve(self, index, value):
#         left = 2 * index + 1
#         right = left + 1
#         if left >= len(self.tree):
#             return index
#         if value <= self.tree[left]:
#             return self._retrieve(left, value)
#         else:
#             return self._retrieve(right, value - self.tree[left])
#
#     def total_priority(self):
#         return self.tree[0]
#
#     def __len__(self):
#         return self.size
#
#
# class NeuralNetwork:
#     def __init__(self, state_shape, action_dim, learning_rate):
#         self.state_shape = state_shape
#         self.action_dim = action_dim
#         self.learning_rate = learning_rate
#
#     def build_model(self):
#         model = Sequential([
#             Conv2D(16, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_shape),
#             Conv2D(32, (4, 4), strides=(2, 2), activation='relu'),
#             Conv2D(32, (3, 3), activation='relu'),
#             Flatten(),
#             Dense(512, activation='relu'),
#             Dense(self.action_dim, activation='linear')
#         ])
#         model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
#         return model
#
#
# class DQNAgent:
#     def __init__(self, state_shape, action_dim, learning_rate=0.001, buffer_capacity=2000, alpha=0.6, beta=0.4):
#         self.state_shape = state_shape
#         self.action_dim = action_dim
#         self.learning_rate = learning_rate
#         self.alpha = alpha  # Priority scaling factor
#         self.beta = beta  # Importance sampling weight factor
#         self.beta_increment_per_sampling = 0.001
#         self.memory = PirorityBuffer(buffer_capacity)
#         self.epsilon = 0.65 #1.00
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.995
#         self.gamma = 0.95
#         self.model = NeuralNetwork(state_shape, action_dim, learning_rate).build_model()
#         self.target_model = NeuralNetwork(state_shape, action_dim, learning_rate).build_model()
#         self.update_target_network()
#         self.max_priority = 1.0
#
#     def update_target_network(self):
#         self.target_model.set_weights(self.model.get_weights())
#
#     def act(self, state):
#         if np.random.rand() <= self.epsilon:
#             # action = np.random.randint(0, self.action_dim)
#             # return 'random', None, action, self.epsilon
#
#             # Use a weighted random choice with uniform distribution
#             action_probs = np.ones(self.action_dim) / self.action_dim
#             action = np.random.choice(self.action_dim, p=action_probs)
#             return 'random', None, action, self.epsilon
#         state = np.expand_dims(state, axis=0)
#         q_values = self.model.predict(state, verbose=0)
#         return 'model', q_values[0], np.argmax(q_values[0]), self.epsilon
#
#         # Add scaled randomness to Q-values
#         # noisy_q_values = q_values[0] + self.epsilon * np.random.randn(self.action_dim)
#         # action = np.argmax(noisy_q_values)
#         # return 'model', q_values[0], action, self.epsilon
#
#     def remember(self, state, action, reward, next_state, done):
#         max_priority = self.max_priority
#         self.memory.add(max_priority, (state, action, reward, next_state, done))
#
#     def replay(self, batch_size):
#         if len(self.memory) < batch_size:
#             return
#         minibatch = []
#         indices = []
#         IS_weights = np.zeros((batch_size, 1))
#         total_priority = self.memory.total_priority()
#         for i in range(batch_size):
#             sample_value = random.uniform(0, total_priority)
#             index, priority, data = self.memory.get(sample_value)
#             states, action, reward, next_state, done = data
#             minibatch.append(data)
#             prob = priority / total_priority
#             IS_weights[i, 0] = (len(self.memory) * prob) ** -self.beta
#             indices.append(index)
#         IS_weights /= IS_weights.max()
#
#         states = np.array([x[0] for x in minibatch])
#         next_states = np.array([x[3] for x in minibatch])
#         targets = self.model.predict(states, verbose=0)
#         next_q_values = self.target_model.predict(next_states, verbose=0)
#
#         for i, (state, action, reward, next_state, done) in enumerate(minibatch):
#             target = reward if done else reward + self.gamma * np.max(next_q_values[i])
#             TD_error = abs(target - targets[i][action])
#             self.memory.update(indices[i], TD_error ** self.alpha)
#             targets[i][action] = target
#             self.max_priority = max(self.max_priority, TD_error)
#         self.model.fit(states, targets, sample_weight=IS_weights.flatten(), epochs=1, verbose=0)
#
#     def decay_epsilon(self):
#         self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
#         self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)


import random
import numpy as np
from keras.src.utils import plot_model
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

class ExperienceReplayBuffer:  # Simple experience replay buffer
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.size = 0
        self.ptr = 0

    def add(self, data):
        if self.size < self.capacity:
            self.buffer.append(data)
            self.size += 1
        else:
            self.buffer[self.ptr] = data
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return self.size


class NeuralNetwork:
    def __init__(self, state_shape, action_dim, learning_rate):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.learning_rate = learning_rate

    def build_model(self):
        model = Sequential([
            Conv2D(16, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_shape),
            Conv2D(32, (4, 4), strides=(2, 2), activation='relu'),
            Conv2D(32, (3, 3), activation='relu'),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(self.action_dim, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        # Add model summary
        model.summary()

        return model


class DQNAgent:
    def __init__(self, state_shape, action_dim, learning_rate=0.01, buffer_capacity=2000):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.memory = ExperienceReplayBuffer(buffer_capacity)
        self.epsilon = 0.65
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.model = NeuralNetwork(state_shape, action_dim, learning_rate).build_model()
        self.target_model = NeuralNetwork(state_shape, action_dim, learning_rate).build_model()
        self.update_target_network()

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.action_dim)
            return 'random', None, action, self.epsilon
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state, verbose=0)
        return 'model', q_values[0], np.argmax(q_values[0]), self.epsilon

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = self.memory.sample(batch_size)

        states = np.array([x[0] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward if done else reward + self.gamma * np.max(next_q_values[i])
            targets[i][action] = target
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        # print(f"[Replay] Loss: {history.history['loss'][0]:.4f}, Epsilon: {self.epsilon:.4f}")
        self.model.fit(states, targets, epochs=1, verbose=0)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # def visualize_weights(self, layer_index=0):
    #     weights = self.model.layers[layer_index].get_weights()[0]
    #     plt.figure(figsize=(15, 15))
    #     for i in range(min(16, weights.shape[-1])):
    #         plt.subplot(4, 4, i + 1)
    #         plt.imshow(weights[:, :, 0, i], cmap='viridis')
    #         plt.axis('off')
    #     plt.show()

    def visualize_model(self, file_path='model_structure.png'):
        plot_model(self.model, to_file=file_path, show_shapes=True, show_layer_names=True)
        print(f"Model structure saved to {file_path}")

    def visualize_weights(self, layer_index=0, save_path=None):
        weights, biases = self.model.layers[layer_index].get_weights()
        print(f"Layer: {self.model.layers[layer_index].name}")
        print(f"Weights shape: {weights.shape}")
        print(f"Biases shape: {biases.shape}")
        num_filters = weights.shape[-1]
        num_channels = weights.shape[-2]

        cols = 8
        rows = (num_filters // cols) + 1
        fig, axes = plt.subplots(rows, cols, figsize=(20, 20))

        for i in range(num_filters):
            f = weights[:, :, :, i]
            if num_channels > 1:
                f = np.mean(f, axis=-1)
            f_min, f_max = f.min(), f.max()
            f = (f - f_min) / (f_max - f_min + 1e-10)
            ax = axes[i // cols, i % cols]
            ax.imshow(f, cmap='viridis')
            ax.axis('off')

        plt.show()
        if save_path:
            fig.savefig(save_path)
            print(f"Weights visualization saved to {save_path}")
