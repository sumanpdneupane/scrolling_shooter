from src.ai_agent.agent_state_and_action import GameActions

class RewardAI:
    def __init__(self):
        self.total_reward = 0
        self.previous_exit_distance = None
        self.jump_cooldown = 0

    def calculate_reward(self, state_dict, state, action, died):
        reward = 0

        current_x = state_dict["player_tile_x"]
        current_y = state_dict["player_tile_y"]
        exit_x = state_dict["exit_tile_x"]
        exit_y = state_dict["exit_tile_y"]

        # --- Major reward/penalty ---
        if died:
            return -100  # Instant penalty

        jump_conditions = (
                state_dict["water_ahead"] or
                state_dict["water_below"] or
                state_dict["space_ahead"] or
                state_dict["on_edge"] or
                not state_dict["path_clear"]
        )

        if jump_conditions:
            if action == GameActions.JUMP:
                reward += 5  # Strong reward for jumping when needed
            else:
                reward -= 3  # Penalty for failing to jump

            # --- Discourage unnecessary jumps ---
        elif action == GameActions.JUMP:
            reward -= 2  # Small penalty for jumping when unnecessary

        # --- Distance reward ---
        current_distance = ((current_x - exit_x) ** 2 + (current_y - exit_y) ** 2) ** 0.5
        if self.previous_exit_distance is not None:
            delta = self.previous_exit_distance - current_distance
            reward += delta * 10  # Reward progress
        self.previous_exit_distance = current_distance

        # --- Essential movement rewards ---
        if action == GameActions.MOVE_RIGHT:
            reward += 10  # Encourage forward motion

        if action == GameActions.JUMP:
            # Only reward jump if there's something to jump over
            if state_dict["water_ahead"] or state_dict["space_ahead"] or not state_dict["path_clear"]:
                reward += 5
            else:
                reward -= 5  # Useless jump

            # Penalize jump spam
            if self.jump_cooldown > 0:
                reward -= 5
            self.jump_cooldown = 3
        else:
            self.jump_cooldown = max(0, self.jump_cooldown - 1)

        # --- Discourage stopping ---
        if action == GameActions.STOP:
            reward -= 5

        # --- Small time penalty ---
        reward -= 0.2

        self.total_reward += reward
        return reward

    def calculate_total_reward(self):
        return self.total_reward

    def reset_total_reward(self):
        self.total_reward = 0
        self.previous_exit_distance = None
        self.jump_cooldown = 0
