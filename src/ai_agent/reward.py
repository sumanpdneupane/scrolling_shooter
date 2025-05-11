from src.ai_agent.agent_state_and_action import GameActions

class RewardAI:
    def __init__(self):
        self.total_reward = 0
        self.previous_exit_distance = None
        self.jump_cooldown = 0

    # def calculate_reward(self, state_dict, state, action, died, enemy_killed=False, shot_fired=False, health_gained= False, ammo_gained=False, grenades_gained=False):
    #     reward = 0
    #
    #     current_x = state_dict["player_tile_x"]
    #     current_y = state_dict["player_tile_y"]
    #     exit_x = state_dict["exit_tile_x"]
    #     exit_y = state_dict["exit_tile_y"]
    #
    #     # --- Major reward/penalty ---
    #     if died:
    #         return -100  # Instant penalty
    #
    #     jump_conditions = (
    #             state_dict["water_ahead"] or
    #             state_dict["water_below"] or
    #             state_dict["space_ahead"] or
    #             state_dict["on_edge"] or
    #             not state_dict["path_clear"]
    #     )
    #
    #     if jump_conditions:
    #         if action == GameActions.JUMP:
    #             reward += 5  # Strong reward for jumping when needed
    #         else:
    #             reward -= 3  # Penalty for failing to jump
    #
    #         # --- Discourage unnecessary jumps ---
    #     elif action == GameActions.JUMP:
    #         reward -= 2  # Small penalty for jumping when unnecessary
    #
    #     # --- Distance reward ---
    #     current_distance = ((current_x - exit_x) ** 2 + (current_y - exit_y) ** 2) ** 0.5
    #     if self.previous_exit_distance is not None:
    #         delta = self.previous_exit_distance - current_distance
    #         reward += delta * 10  # Reward progress
    #     self.previous_exit_distance = current_distance
    #
    #     # --- Essential movement rewards ---
    #     if action == GameActions.MOVE_RIGHT:
    #         reward += 10  # Encourage forward motion
    #         # Bonus: Moving forward on a clear path
    #         if state_dict["path_clear"]:
    #             reward += 5  # Extra reward for efficient forward motion
    #     elif action == GameActions.MOVE_LEFT:
    #         reward -= 2  # Mild penalty for going backward
    #
    #     if not state_dict["path_clear"] and action == GameActions.MOVE_RIGHT:
    #         reward -= 3  # Penalty for trying to move forward into a blocked path
    #
    #     if action == GameActions.JUMP:
    #         # Only reward jump if there's something to jump over
    #         if state_dict["water_ahead"] or state_dict["space_ahead"] or not state_dict["path_clear"]:
    #             reward += 5
    #         else:
    #             reward -= 5  # Useless jump
    #
    #         # Penalize jump spam
    #         if self.jump_cooldown > 0:
    #             reward -= 5
    #         self.jump_cooldown = 3
    #     else:
    #         self.jump_cooldown = max(0, self.jump_cooldown - 1)
    #
    #     # --- Discourage stopping ---
    #     if action == GameActions.STOP:
    #         reward -= 5
    #
    #     # --- Reward for shooting ---
    #     if shot_fired:
    #         if state_dict["enemy_distance"] < 0.3:  # Close enemy
    #             reward += 2
    #         else:
    #             reward -= 1  # Wasteful shot
    #
    #     # --- Reward for killing enemy ---
    #     if enemy_killed:
    #         reward += 10
    #
    #     # --- Reward for gaining health ---
    #     if health_gained:
    #         reward += 10  # Reward for healing
    #
    #     # --- Reward for gaining ammo ---
    #     if ammo_gained:
    #         reward += 5  # Adjust as needed
    #
    #     # --- Reward for gaining grenades ---
    #     if grenades_gained:
    #         reward += 7  # Adjust as needed
    #
    #     # --- Small time penalty ---
    #     reward -= 0.2
    #
    #     self.total_reward += reward
    #     return reward

    def calculate_reward(self, state_dict, state, action, died, enemy_killed=False, shot_fired=False, health_gained= False, ammo_gained=False, grenades_gained=False, reached_exit=False):
        reward = 0

        # Major penalties/rewards
        if died: return -100
        if reached_exit: return 300

        # Directional exit movement
        exit_dir = 1 if state_dict["exit_tile_x"] > 0 else -1
        if action == GameActions.MOVE_RIGHT:
            reward += 5 * exit_dir
        elif action == GameActions.MOVE_LEFT:
            reward += 5 * (-exit_dir)

        # Distance-based reward
        current_dist = state_dict["exit_distance"]
        if self.previous_exit_distance:
            delta = self.previous_exit_distance - current_dist
            reward += delta * 20 + (1 if delta > 0 else 0)

        # Jump optimization
        if action == GameActions.JUMP:
            if any([state_dict["water_ahead"], state_dict["on_edge"]]):
                reward += 8
            else:
                reward -= 4

        # Shooting logic
        if shot_fired:
            if state_dict["enemy_distance"] < 0.3:
                reward += 3 if enemy_killed else -3
            else:
                reward -= 3

        # Resource management
        reward += 5 * 1 if health_gained else 0
        reward += 2 * 1 if ammo_gained else 0
        reward += 3 * 1 if grenades_gained else 0

        # Time penalty
        reward -= 0.02

        self.total_reward += reward

        return reward
    def calculate_total_reward(self):
        return self.total_reward

    def reset_total_reward(self):
        self.total_reward = 0
        self.previous_exit_distance = None
        self.jump_cooldown = 0
