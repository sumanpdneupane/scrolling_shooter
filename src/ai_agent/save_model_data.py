import os

import cv2
from tensorflow import keras
from tensorflow.python.keras import Sequential

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

        if os.path.exists(self.model_path):
            loaded_model = keras.models.load_model(self.model_path)

            # Set weights of the internal Keras model
            q_network.model.set_weights(loaded_model.get_weights())
            target_network.model.set_weights(loaded_model.get_weights())

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
        return None  # default if no file exists

    def save_current_frame(self, current_frame, iteration):
        # Create the directory if it doesn't exist
        save_directory = "src/data_logs/extract_image"
        os.makedirs(save_directory, exist_ok=True)

        # Save the image
        image_path = os.path.join(save_directory, f"frame_{iteration}.png")
        cv2.imwrite(image_path, current_frame)