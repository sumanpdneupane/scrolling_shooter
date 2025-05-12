import torch
import os

# ----------------------------- SAVE/LOAD MODEL: SaveFutureLearning ----------------------------- #
# class SaveFutureLearning:
#     def __init__(self, model_path, epsilon_path, episode_path):
#         self.model_path = model_path
#         self.epsilon_path = epsilon_path
#         self.episode_path = episode_path
#
#     def save_model(self, model, agent, episode):
#         # Save model weights
#         torch.save(model.state_dict(), self.model_path)
#
        # # Save epsilon value
        # with open(self.epsilon_path, "w") as f:
        #     f.write(str(agent.epsilon))
        #
        # # Save episode number
        # with open(self.episode_path, "w") as f:
        #     f.write(str(episode))
#
#     def load_model(self, q_network, target_network, agent):
#         # Load the Q-network model from the model path, Load model weights if exists
#         if os.path.exists(self.model_path):
#             q_network.load_state_dict(torch.load(self.model_path))
#             target_network.load_state_dict(q_network.state_dict())
#             # print("Loaded saved model.")
#
#             # Load epsilon if file is not empty
#             if os.path.exists(self.epsilon_path):
#                 with open(self.epsilon_path, "r") as f:
#                     content = f.read().strip()
#                     if content:
#                         agent.epsilon = float(content)
#                         # print("Loaded saved epsilon.")
#                     else:
#                         print("Epsilon file is empty. Using default epsilon.")
#
#     def load_episode(self):
#         # Load episode number if exists
#         if os.path.exists(self.episode_path):
#             with open(self.episode_path, "r") as f:
#                 return int(f.read())
#         return 0  # default if no file exists

import os
import numpy as np
from tensorflow import keras
from tensorflow.python.keras import Sequential

from src.ai_agent.agent import NeuralNetwork, DQNAgent


class SaveFutureLearning:
    def __init__(self, model_path, epsilon_path, episode_path):
        self.model_path = model_path
        self.epsilon_path = epsilon_path
        self.episode_path = episode_path

    def save_model(self, model, agent, episode):
        # Save entire Keras model
        model.save(self.model_path)

        # Save epsilon value
        with open(self.epsilon_path, "w") as f:
            f.write(str(agent.epsilon))

        # Save episode number
        with open(self.episode_path, "w") as f:
            f.write(str(episode))

    def load_model(self, q_network: Sequential, target_network: Sequential):
        # print("---------Before: q_network: ", q_network.get_weights(), "target_network: ", target_network.get_weights())

        # Load Keras model if exists
        if os.path.exists(self.model_path):
            loaded_model = keras.models.load_model(self.model_path)
            target_network.set_weights(loaded_model.get_weights())
            q_network.set_weights(loaded_model.get_weights())

            # print("---------After: q_network: ", q_network.get_weights(), "target_network: ", target_network.get_weights())

        return q_network, target_network

    def load_episode(self):
        # # Load episode number if exists
        # if os.path.exists(self.episode_path):
        #     with np.load(self.episode_path) as data:
        #         return int(data['episode'])
        # return 0
        # Load episode number if exists
        if os.path.exists(self.episode_path):
            with open(self.episode_path, "r") as f:
                return int(f.read())
        return 0  # default if no file exists

    def load_epsilon(self):
        # Load epsilon number if exists
        if os.path.exists(self.epsilon_path):
            with open(self.epsilon_path, "r") as f:
                return float(f.read())
        return 0.62  # default if no file exists