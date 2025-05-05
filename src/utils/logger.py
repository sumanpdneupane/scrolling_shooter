import os
from pathlib import Path

from matplotlib import pyplot as plt

from src.graph.live_plotter import LivePlotter
from src.settings import *

# ----------------------------- LOGGER ----------------------------- #
class TrainingLogger:
    def __init__(self, filename= TRAINING_LOG_PATH):
        self.filename = filename
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Total Reward", "Epsilon"])

    def log(self, episode, reward, epsilon):
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward, epsilon])

class GraphLogger(LivePlotter):
    def __init__(self):
        super().__init__()
        # Create directories if they don't exist
        self.plot_dir = Path("src/data_logs/graph")
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            'episodes': [],
            'total_rewards': [],
            'average_rewards': []
        }

    def log_historical(self):
        plt.figure()
        plt.plot(self.history['episodes'], self.history['total_rewards'], label='Total Reward')
        plt.plot(self.history['episodes'], self.history['average_rewards'], label='Average Reward')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.legend()

        # Ensure directory exists each time
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.plot_dir / 'training_history.png')
        plt.close()