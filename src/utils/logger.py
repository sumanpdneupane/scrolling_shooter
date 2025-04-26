import os
import csv

# ----------------------------- LOGGER ----------------------------- #
class TrainingLogger:
    def __init__(self, filename="training_log.csv"):
        self.filename = filename
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Total Reward", "Epsilon"])

    def log(self, episode, reward, epsilon):
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward, epsilon])