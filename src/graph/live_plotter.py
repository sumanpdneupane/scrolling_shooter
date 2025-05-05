from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import threading
from src.ai_agent.agent_state_and_action import GameActions


class LivePlotter:
    def __init__(self):
        # Create directories if they don't exist
        self.plot_dir = Path("src/data_logs/graph")
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        self.rewards = []
        self.epsilons = []
        self.actions = {action.name: 0 for action in GameActions}
        self.update_interval = 5  # Update every 5 episodes

        # Configure plots
        self.ax1.set_title('Training Progress')
        self.ax1.set_ylabel('Reward')
        self.ax2.set_xlabel('Episodes')
        self.ax2.set_ylabel('Epsilon')

        # Start animation thread
        self.anim = FuncAnimation(self.fig, self.update_plot, interval=1000)
        self.thread = threading.Thread(target=plt.show)
        self.thread.daemon = True
        self.thread.start()

    def log(self, episode, reward, epsilon, action_counts):
        self.rewards.append(reward)
        self.epsilons.append(epsilon)

        # Update action distribution
        for action, count in action_counts.items():
            self.actions[action.name] += count

    def update_plot(self, frame):
        self.ax1.clear()
        self.ax2.clear()

        # Plot rewards
        self.ax1.plot(self.rewards, 'b-', label='Episode Reward')
        self.ax1.plot(self._smooth(self.rewards), 'r-', label='Smoothed Reward')
        self.ax1.legend()

        # Plot epsilon decay
        self.ax2.plot(self.epsilons, 'g-', label='Exploration Rate')
        self.ax2.legend()

        # Plot action distribution every 10 updates
        if len(self.rewards) % 10 == 0:
            self.plot_action_distribution()

    def _smooth(self, values, weight=0.9):
        smoothed = []
        last = values[0]
        for point in values:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    def plot_action_distribution(self):
        action_fig = plt.figure()
        plt.bar(self.actions.keys(), self.actions.values())
        plt.title('Action Distribution')
        plt.ylabel('Count')

        # Ensure directory exists each time
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.plot_dir / f'action_distribution_{len(self.rewards)}.png')
        plt.close(action_fig)