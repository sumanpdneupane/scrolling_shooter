import torch
import os

# ----------------------------- SAVE/LOAD MODEL: SaveFutureLearning ----------------------------- #
class SaveFutureLearning:
    def __init__(self, model_path, epsilon_path, episode_path):
        self.model_path = model_path
        self.epsilon_path = epsilon_path
        self.episode_path = episode_path

    def save_model(self, model, agent):
        # Save model weights
        torch.save(model.state_dict(), self.model_path)

        # Save epsilon value
        with open(self.epsilon_path, "w") as f:
            f.write(str(agent.epsilon))

        # Save episode number
        with open(self.episode_path, "w") as f:
            f.write(str(agent.episode))

    def load_model(self, q_network, target_network, agent):
        # Load the Q-network model from the model path, Load model weights if exists
        if os.path.exists(self.model_path):
            q_network.load_state_dict(torch.load(self.model_path))
            target_network.load_state_dict(q_network.state_dict())
            # print("Loaded saved model.")

            # Load epsilon if file is not empty
            if os.path.exists(self.epsilon_path):
                with open(self.epsilon_path, "r") as f:
                    content = f.read().strip()
                    if content:
                        agent.epsilon = float(content)
                        # print("Loaded saved epsilon.")
                    else:
                        print("Epsilon file is empty. Using default epsilon.")

    def load_episode(self):
        # Load episode number if exists
        if os.path.exists(self.episode_path):
            with open(self.episode_path, "r") as f:
                return int(f.read())
        return 0  # default if no file exists