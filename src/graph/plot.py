import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class TrainingVisualizer:
    def __init__(self, filename="src/data_logs/graph_plot_result/training_data.csv"):
        self.filename = filename
        self._init_csv_file()

    def _init_csv_file(self):
        """Initialize CSV file with headers"""
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'episode', 'total_reward',
                    'success', 'epsilon', 'steps',
                    'time_taken', 'distance_traveled'
                ])

    def save_episode(self, episode, total_reward, success, epsilon, steps, time_taken, distance_traveled):
        """Save episode data to CSV"""
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                episode,
                total_reward,
                int(success),  # Store as 0/1 for easier analysis
                epsilon,
                steps,
                time_taken,  # In seconds
                distance_traveled  # In pixels or game units
            ])

    def load_data(self):
        """Load historical data from CSV"""
        return pd.read_csv(self.filename, parse_dates=['timestamp'])

    def plot_progress(self, window=100):
        """Generate plots from saved data"""
        df = self.load_data()

        # Compute speed if columns exist
        if 'time_taken' in df and 'distance_traveled' in df:
            df['speed'] = df['distance_traveled'] / df['time_taken']
        else:
            df['speed'] = None  # Handle missing data
        # if 'time_taken' in df and 'distance_traveled' in df:
        #     df['speed'] = df['distance_traveled'] / df['time_taken']

        plt.figure(figsize=(18, 15))  # Adjusted figure size for more plots

        # Reward Plot
        plt.subplot(3, 2, 1)
        sns.lineplot(data=df, x='episode', y='total_reward')
        plt.title('Reward Progression')
        plt.grid(True)

        # Success Rate Plot
        plt.subplot(3, 2, 2)
        df['rolling_success'] = df['success'].rolling(window).mean()
        sns.lineplot(data=df, x='episode', y='rolling_success')
        plt.title(f'Success Rate (Last {window} Episodes)')
        plt.grid(True)

        # Epsilon Decay Plot
        plt.subplot(3, 2, 3)
        sns.lineplot(data=df, x='episode', y='epsilon')
        plt.title('Exploration Rate (Epsilon)')
        plt.grid(True)

        # Steps per Episode
        plt.subplot(3, 2, 4)
        sns.lineplot(data=df, x='episode', y='steps')
        plt.title('Steps per Episode')
        plt.grid(True)

        # Speed Progression (if available)
        if 'speed' in df:
            plt.subplot(3, 2, 5)
            sns.lineplot(data=df, x='episode', y='speed')
            plt.title('Average Speed Development')
            plt.grid(True)

            # Speed vs Success
            plt.subplot(3, 2, 6)
            sns.boxplot(data=df, x='success', y='speed')
            plt.title('Speed Comparison: Successful vs Failed Episodes')
            plt.grid(True)

        plt.tight_layout()
        plt.savefig('src/data_logs/graph_plot_result/training_progress.png')
        plt.show()

    def plot_combined(self, window=100):
        """Combined reward, success rate, and speed plot"""
        df = self.load_data()
        df['rolling_success'] = df['success'].rolling(window).mean()

        # # Compute speed if possible
        # if 'time_taken' in df and 'distance_traveled' in df:
        #     df['speed'] = df['distance_traveled'] / df['time_taken']
        # else:
        #     df['speed'] = None  # Handle missing data

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Primary Y-axis for Reward and Speed
        color_reward = 'tab:blue'
        ax1.set_xlabel('Episode')
        # ax1.set_ylabel('Reward / Speed', color=color_reward)
        ax1.set_ylabel('Reward', color=color_reward)
        sns.lineplot(data=df, x='episode', y='total_reward', ax=ax1, color=color_reward, label='Reward')
        # if df['speed'].notnull().all():
        #     sns.lineplot(data=df, x='episode', y='speed', ax=ax1, color='tab:green', label='Speed')
        ax1.tick_params(axis='y', labelcolor=color_reward)
        ax1.legend(loc='upper left')

        # Secondary Y-axis for Success Rate
        ax2 = ax1.twinx()
        color_success = 'tab:red'
        ax2.set_ylabel('Success Rate', color=color_success)
        sns.lineplot(data=df, x='episode', y='rolling_success', ax=ax2, color=color_success, label='Success Rate')
        ax2.tick_params(axis='y', labelcolor=color_success)

        # plt.title('Training Progress: Reward, Speed, and Success Rate')
        plt.title('Training Progress: Reward and Success Rate')
        plt.grid(True)
        plt.savefig('src/data_logs/graph_plot_result/combined_progress.png')
        plt.show()
