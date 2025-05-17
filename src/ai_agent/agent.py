import tensorflow as tf
from keras.src.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, BatchNormalization, \
    Dropout
from tensorflow.keras.models import Model
import numpy as np
import random
from collections import deque



class DualInputQNetwork:
    def __init__(self, image_shape, extra_features_dim, num_actions):
        self.image_shape = image_shape
        self.extra_features_dim = extra_features_dim
        self.num_actions = num_actions
        self.model = self._build_model()

    def _build_model(self):
        image_input = Input(shape=self.image_shape, name="image_input")

        # First convolutional block
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
        # x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)  # (160, 200, 32)

        # Second convolutional block
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        # x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)  # (80, 100, 64)

        # Third convolutional block
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        # x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)  # (40, 50, 64)

        # Flatten the output for dense layers
        x = Flatten()(x)

        # Extra features input
        extra_input = Input(shape=(self.extra_features_dim,), name="extra_input")
        y = Dense(8, activation='relu')(extra_input)

        # Combine image features and extra features
        combined = Concatenate()([x, y])

        # Dense layers with Dropout for regularization
        z = Dense(32, activation='relu')(combined)
        z = Dropout(0.5)(z)  # 50% Dropout
        z = Dense(16, activation='relu')(z)
        z = Dropout(0.5)(z)  # 50% Dropout

        # Output layer with linear activation for Q-values
        output = Dense(self.num_actions, activation='linear')(z)

        # Build and compile the model
        model = Model(inputs=[image_input, extra_input], outputs=output)
        optimizer = Adam(clipvalue=1.0)  # Clip gradients to prevent large updates
        model.compile(loss='mse', optimizer=optimizer)

        return model

    def predict(self, image_state, extra_features):
        return self.model.predict([image_state, extra_features], verbose=0)

    def train(self, image_states, extra_features, target_q_values):
        return self.model.train_on_batch([image_states, extra_features], target_q_values)

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = tf.keras.models.load_model(filepath)


# class DQNAgent:
#     def __init__(
#         self,
#         image_shape,
#         extra_features_dim,
#         num_actions,
#         memory_size=10000,
#         batch_size=64,
#         gamma=0.99,
#         epsilon_start=0.50,
#         epsilon_min=0.1,
#         epsilon_decay=0.995,
#         target_update_freq=1000
#     ):
#         self.step_count = 0
#         self.image_shape = image_shape
#         self.extra_features_dim = extra_features_dim
#         self.num_actions = num_actions
#
#         self.memory = deque(maxlen=memory_size)
#         self.batch_size = batch_size
#         self.gamma = gamma
#
#         self.epsilon = epsilon_start
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.epsilon_decay_steps = 50000
#
#         self.learn_step_counter = 0
#         self.target_update_freq = target_update_freq
#
#         self.q_value_history = []
#         self.q_stagnant_threshold = 10  # Number of steps to detect stagnation
#         self.q_similarity_tolerance = 1e-2  # Tolerance for similarity
#         self.rewards_buffer = deque(maxlen=50)
#         self.death_positions = deque()
#
#         # Add new buffer to store the past 10 experiences (death positions, rewards, player positions)
#         # self.death_positions_rewards = deque(maxlen=10)  # stores tuples (death_position, reward, player_position)
#
#         # Main and target networks
#         self.q_network = DualInputQNetwork(image_shape, extra_features_dim, num_actions)
#         self.target_network = DualInputQNetwork(image_shape, extra_features_dim, num_actions)
#         self.update_target_network()
#
#     def save_reward(self, reward, action, player_position=None):
#         # Maintain a buffer of the last 50 rewards
#         self.rewards_buffer.append((reward, action, player_position))
#         if len(self.rewards_buffer) > 50:
#             self.rewards_buffer.pop()
#
#     def past_death_list(self):
#         past_deaths = []
#
#         # print("rewards_buffer:", self.rewards_buffer)
#
#         # Iterate through the rewards buffer to find rewards < 0 (death)
#         for i in range(1, len(self.rewards_buffer)):  # Start from index 1 to get previous positions
#             reward,_, death_position = self.rewards_buffer[i]
#
#             # If the reward is negative (death)
#             # if reward < 0:
#             # Get the last 10 player positions before the death, excluding the death position itself
#             start_index = max(0, i - 10)  # Ensure we don't go below index 0
#             last_10_positions = []
#
#             # Collect positions from rewards_buffer[start_index] to rewards_buffer[i-1]
#             for j in range(start_index, i):  # Iterate over previous rewards
#                 _, _, pos = self.rewards_buffer[j]  # Get the position
#                 last_10_positions.append(pos)
#
#             # Append the death position at the end of the list
#             last_10_positions.append(death_position)
#
#             # Add the last 10 positions leading up to the death (with death position at the end)
#             past_deaths.append(tuple(last_10_positions))
#
#         self.death_positions.append(past_deaths)  # Append to death_positions
#
#
#     def get_recent_rewards(self):
#         return self.rewards_buffer
#
#     def update_target_network(self):
#         self.target_network.model.set_weights(self.q_network.model.get_weights())
#
#     def remember(self, state_img, extra_features, action, reward, next_state_img, next_extra_features, done):
#         self.memory.append((state_img, extra_features, action, reward, next_state_img, next_extra_features, done))
#
#     def softmax(self, q_values, temperature=1.0):
#         """Apply softmax to Q-values to get a probability distribution."""
#         q_values = np.array(q_values)
#         q_values = q_values / temperature
#         e_x = np.exp(q_values - np.max(q_values))  # stability
#         return e_x / e_x.sum()
#
#     def act(self, state_img, extra_features, player_position, temperature=0.3, mode="hardmax"):
#         q_values = self.q_network.predict(state_img[np.newaxis, ...], extra_features[np.newaxis, ...])[0]
#         # is_stagnant = self.is_q_value_stagnant(q_values)
#
#         # Clip Q-values to be non-negative
#         q_values = np.maximum(q_values, 0)  # Ensure no negative values
#         if np.random.rand() <= self.epsilon:
#             action_type = "explore"
#             action = random.randint(0, self.num_actions - 1)
#         # elif is_stagnant:
#         #     action_type = "explore_stagnation"
#         #     action = random.randint(0, self.num_actions - 1)
#
#         else:
#             if mode == "softmax":
#                 if np.max(q_values) <= 0 or np.all(q_values == q_values[0]):
#                     action_type = "explore-s-fallback"
#                     action = random.randint(0, self.num_actions - 1)
#                 else:
#                     action_type = "exploit_softmax"
#                     probs = self.softmax(q_values, temperature)
#                     action = np.random.choice(self.num_actions, p=probs)
#             elif mode == "hardmax":
#                 if np.max(q_values) <= 0 or np.all(q_values == q_values[0]):
#                     action_type = "explore-h-fallback"
#                     action = random.randint(0, self.num_actions - 1)
#                 else:
#                     action_type = "exploit"
#                     action = np.argmax(q_values)
#             else:
#                 raise ValueError("Unsupported action selection mode. Use 'softmax' or 'hardmax'.")
#
#         return action_type, q_values, action, self.epsilon
#
#     def is_q_value_stagnant(self, current_q_values):
#         self.q_value_history.append(current_q_values)
#
#         if len(self.q_value_history) > self.q_stagnant_threshold:
#             self.q_value_history.pop(0)
#
#         if len(self.q_value_history) < self.q_stagnant_threshold:
#             return False
#
#         # Check variance across the Q-values
#         q_stack = np.stack(self.q_value_history)
#         max_change = np.max(np.abs(q_stack - q_stack[0]))
#
#         print(f"[DEBUG] Q-Value max change over last {self.q_stagnant_threshold} steps: {max_change:.6f}")
#
#         return max_change < self.q_similarity_tolerance
#
#     def replay(self):
#         if len(self.memory) < self.batch_size:
#             return
#
#         minibatch = random.sample(self.memory, self.batch_size)
#
#         # Extract batch data
#         state_imgs = np.array([sample[0] for sample in minibatch])
#         extra_feats = np.array([sample[1] for sample in minibatch])
#         actions = [sample[2] for sample in minibatch]
#         rewards = [sample[3] for sample in minibatch]
#         next_state_imgs = np.array([sample[4] for sample in minibatch])
#         next_extra_feats = np.array([sample[5] for sample in minibatch])
#         dones = [sample[6] for sample in minibatch]
#
#         # Batch predictions
#         q_current_batch = self.q_network.predict(state_imgs, extra_feats)
#         q_next_batch = self.target_network.predict(next_state_imgs, next_extra_feats)
#
#         targets = np.copy(q_current_batch)
#
#         for i in range(self.batch_size):
#             if dones[i]:
#                 targets[i][actions[i]] = rewards[i]
#             else:
#                 targets[i][actions[i]] = rewards[i] + self.gamma * np.max(q_next_batch[i])
#
#         self.q_network.train(state_imgs, extra_feats, targets)
#
#         # Update target network
#         self.learn_step_counter += 1
#         if self.learn_step_counter % self.target_update_freq == 0:
#             self.update_target_network()
#
#     def update_epsilon(self, rate = 1.0):
#         if any(reward < 0 for reward,_, _ in self.rewards_buffer):
#             self.epsilon = 0.20
#             return
#         self.step_count += 1
#         self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
#         # decay_rate = ((self.epsilon - self.epsilon_min) / self.epsilon_decay_steps) * rate
#         # self.epsilon = max(self.epsilon_min, self.epsilon - decay_rate * self.step_count)
#
#     def save(self, filepath):
#         self.q_network.save(filepath)
#
#     def load(self, filepath):
#         self.q_network.load(filepath)
#         self.update_target_network()

class DQNAgent:
    def __init__(
        self,
        image_shape,
        extra_features_dim,
        num_actions,
        memory_size=10000,
        batch_size=64,
        gamma=0.99,
        epsilon_start=0.50,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        target_update_freq=100
    ):
        self.step_count = 0
        self.image_shape = image_shape
        self.extra_features_dim = extra_features_dim
        self.num_actions = num_actions

        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.learn_step_counter = 0
        self.target_update_freq = target_update_freq

        self.q_value_history = []
        self.q_stagnant_threshold = 10  # Number of steps to detect stagnation
        self.q_similarity_tolerance = 1e-2  # Tolerance for similarity
        self.rewards_buffer = deque(maxlen=50)
        self.death_positions = deque()

        # Main and target networks
        self.q_network = DualInputQNetwork(image_shape, extra_features_dim, num_actions)
        self.target_network = DualInputQNetwork(image_shape, extra_features_dim, num_actions)
        self.update_target_network()

    def save_reward(self, reward, action, player_position=None):
        # Maintain a buffer of the last 50 rewards
        self.rewards_buffer.append((reward, action, player_position))
        if len(self.rewards_buffer) > 50:
            self.rewards_buffer.pop()

    def past_death_list(self):
        past_deaths = []

        for i in range(1, len(self.rewards_buffer)):  # Start from index 1 to get previous positions
            reward, _, death_position = self.rewards_buffer[i]

            if reward < 0:  # Only process if the reward is negative (indicating death)
                start_index = max(0, i - 10)
                last_10_positions = [pos for _, _, pos in self.rewards_buffer[start_index:i]]
                last_10_positions.append(death_position)

                past_deaths.append(tuple(last_10_positions))

        self.death_positions.append(past_deaths)  # Append to death_positions

    def get_recent_rewards(self):
        return self.rewards_buffer

    def update_target_network(self):
        self.target_network.model.set_weights(self.q_network.model.get_weights())

    def remember(self, state_img, extra_features, action, reward, next_state_img, next_extra_features, done):
        self.memory.append((state_img, extra_features, action, reward, next_state_img, next_extra_features, done))

    def softmax(self, q_values, temperature=1.0):
        """Apply softmax to Q-values to get a probability distribution."""
        q_values = np.array(q_values)
        q_values = q_values / temperature
        e_x = np.exp(q_values - np.max(q_values))  # stability
        return e_x / e_x.sum()

    def act(self, state_img, extra_features, temperature=0.3, mode="hardmax"):
        q_values = self.q_network.predict(state_img[np.newaxis, ...], extra_features[np.newaxis, ...])[0]

        # Check if the Q-values are stagnant
        # is_stagnant = self.is_q_value_stagnant(q_values)
        q_values = np.maximum(q_values, 0)  # Ensure no negative values

        # if allow_jump or collide:
        #     action_type = "explore"
        #     action = GameActions.JUMP
        # else:
        #     action_type = "explore"
        #     action = GameActions.MOVE_RIGHT

        if np.random.rand() <= self.epsilon:
            action_type = "explore"
            action = random.randint(0, self.num_actions - 1)
        else:
            if mode == "softmax":
                if np.max(q_values) <= 0 or np.all(q_values == q_values[0]):
                    action_type = "explore-s-fallback"
                    action = random.randint(0, self.num_actions - 1)
                else:
                    action_type = "exploit_softmax"
                    probs = self.softmax(q_values, temperature)
                    action = np.random.choice(self.num_actions, p=probs)
            elif mode == "hardmax":
                if np.max(q_values) <= 0 or np.all(q_values == q_values[0]):
                    action_type = "explore-h-fallback"
                    action = random.randint(0, self.num_actions - 1)
                else:
                    action_type = "exploit"
                    action = np.argmax(q_values)
            else:
                raise ValueError("Unsupported action selection mode. Use 'softmax' or 'hardmax'.")

        return action_type, q_values, action, self.epsilon

    def is_q_value_stagnant(self, current_q_values):
        self.q_value_history.append(current_q_values)

        if len(self.q_value_history) > self.q_stagnant_threshold:
            self.q_value_history.pop(0)

        if len(self.q_value_history) < self.q_stagnant_threshold:
            return False

        # Check variance across the Q-values
        q_stack = np.stack(self.q_value_history)
        max_change = np.max(np.abs(q_stack - q_stack[0]))

        print(f"[DEBUG] Q-Value max change over last {self.q_stagnant_threshold} steps: {max_change:.6f}")

        return max_change < self.q_similarity_tolerance

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        state_imgs = np.array([sample[0] for sample in minibatch])
        extra_feats = np.array([sample[1] for sample in minibatch])
        actions = [sample[2] for sample in minibatch]
        rewards = [sample[3] for sample in minibatch]
        next_state_imgs = np.array([sample[4] for sample in minibatch])
        next_extra_feats = np.array([sample[5] for sample in minibatch])
        dones = [sample[6] for sample in minibatch]

        q_current_batch = self.q_network.predict(state_imgs, extra_feats)
        q_next_batch = self.target_network.predict(next_state_imgs, next_extra_feats)

        targets = np.copy(q_current_batch)

        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(q_next_batch[i])

        self.q_network.train(state_imgs, extra_feats, targets)

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.update_target_network()

    def update_epsilon(self):
        # if self.epsilon <= 0.1:
        #     self.epsilon = 0.25
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        self.q_network.save(filepath)

    def load(self, filepath):
        self.q_network.load(filepath)
        self.update_target_network()


