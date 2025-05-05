from src.ai_agent.agent_state_and_action import GameActions

# class RewardAI:
#     def __init__(self):
#         self.total_reward = 0
#         self.previous_x_position = None
#         self.previous_exit_distance = None
#         self.step_count = 0
#         self.last_health = 100
#         self.consecutive_jumps = 0
#         self.jump_cooldown = 0
#
#     def calculate_reward(self, state_dict, state, action, died):
#         self.step_count += 1
#         reward = 0
#
#         current_x = state_dict["player_tile_x"]
#         current_y = state_dict["player_tile_y"]
#         exit_tile_x = state_dict["exit_tile_x"]
#         exit_tile_y = state_dict["exit_tile_y"]
#         current_health = state_dict["health"] * 100
#
#         current_distance = ((current_x - exit_tile_x) ** 2 + (current_y - exit_tile_y) ** 2) ** 0.5
#         correct_direction = 1 if exit_tile_x > current_x else 0
#
#         # === MAJOR REWARDS/PENALTIES ===
#         if died:
#             reward -= 100
#             print("DIED! Penalty: -100")
#
#         if state_dict["near_exit"]:
#             reward += 100
#             print("NEAR EXIT! Reward: +100")
#
#         if self.last_health is not None:
#             health_change = current_health - self.last_health
#             if health_change < 0:
#                 reward += health_change * 5
#                 print(f"HEALTH LOSS: {health_change}, Penalty: {health_change * 5}")
#
#         if action == GameActions.SHOOT:
#             reward += 2
#         if action == GameActions.GRENADE:
#             reward += 3
#
#
#         # === MOVEMENT REWARDS ===
#         if action == GameActions.MOVE_RIGHT:
#             reward += 30  # Reduced from 50
#             if state_dict.get("on_ground", True):
#                 reward += 10
#                 print("GROUND MOVEMENT RIGHT! Bonus: +10")
#             print("MOVING RIGHT! Reward: +30")
#         elif action == GameActions.MOVE_LEFT:
#             reward += 3  # Reduced from 5
#             if state_dict.get("on_ground", True):
#                 reward += 2
#                 print("GROUND MOVEMENT LEFT! Bonus: +2")
#             print("MOVING LEFT! Reward: +3")
#
#         if action == correct_direction:
#             reward += 30
#             print("CORRECT DIRECTION towards exit! Bonus: +30")
#
#         if self.previous_exit_distance is not None:
#             distance_change = self.previous_exit_distance - current_distance
#             if distance_change > 0:
#                 progress_reward = distance_change * 10
#                 reward += progress_reward
#                 print(f"PROGRESS TOWARDS EXIT: {distance_change:.2f}, Reward: +{progress_reward:.2f}")
#             else:
#                 penalty = distance_change * 5
#                 reward += penalty
#                 print(f"MOVED AWAY FROM EXIT: {distance_change:.2f}, Penalty: {penalty:.2f}")
#
#         # === SITUATIONAL REWARDS ===
#         if state_dict["water_ahead"] or state_dict["water_below"]:
#             if action == GameActions.JUMP:
#                 reward += 5
#                 print("GOOD JUMP over water! Reward: +5")
#             else:
#                 reward -= 3
#                 print("FAILED TO JUMP over water! Penalty: -3")
#
#         if state_dict["space_ahead"]:
#             if action == GameActions.JUMP:
#                 reward += 5
#                 print("GOOD JUMP over gap! Reward: +5")
#             else:
#                 reward -= 3
#                 print("FAILED TO JUMP over gap! Penalty: -3")
#
#         if not state_dict["path_clear"]:
#             if action == GameActions.JUMP:
#                 reward += 5
#                 print("GOOD JUMP over obstacle! Reward: +5")
#             else:
#                 reward -= 3
#                 print("FAILED TO JUMP over obstacle! Penalty: -3")
#
#         # === ENHANCED JUMP CONTROL ===
#         ON_GROUND = state_dict.get("on_ground", True)
#         AIRBORNE = not ON_GROUND
#
#         # Track consecutive jumps
#         if action == GameActions.JUMP:
#             self.consecutive_jumps += 1
#         else:
#             self.consecutive_jumps = 0
#
#         # Penalty for unnecessary jump
#         jump_penalty = 0
#         if action == GameActions.JUMP and state_dict["path_clear"] and not state_dict["water_ahead"] and not state_dict["space_ahead"] and ON_GROUND:
#             jump_penalty = -5
#
#         if jump_penalty < 0 and self.consecutive_jumps > 1:
#             jump_penalty *= (1 + self.consecutive_jumps * 0.5)
#             print(f"CONSECUTIVE UNNECESSARY JUMPS ({self.consecutive_jumps}): Penalty {jump_penalty:.2f}")
#
#         if action == GameActions.JUMP and AIRBORNE:
#             reward -= 4
#             print("MID-AIR JUMP! Penalty: -4")
#
#         reward += jump_penalty
#
#         # Reward for staying grounded
#         if ON_GROUND and action != GameActions.JUMP and not state_dict["water_ahead"] and not state_dict["space_ahead"]:
#             reward += 2
#
#         # Cooldown mechanism for jump spam
#         if action == GameActions.JUMP:
#             if self.jump_cooldown > 0:
#                 cooldown_penalty = 3 * self.jump_cooldown
#                 reward -= cooldown_penalty
#                 print(f"JUMP SPAM! Cooldown penalty: -{cooldown_penalty}")
#             self.jump_cooldown = 3
#         else:
#             self.jump_cooldown = max(0, self.jump_cooldown - 1)
#
#         # === OTHER PENALTIES ===
#         if action == GameActions.STOP:
#             reward -= 5
#             print("STOPPED! Penalty: -5")
#
#         reward -= 0.2  # Time penalty
#
#         if self.step_count % 100 == 0:
#             reward += 1  # Exploration bonus
#
#         # === STATE UPDATES ===
#         self.previous_x_position = current_x
#         self.previous_exit_distance = current_distance
#         self.last_health = current_health
#
#         print(f"Action: {action}, Final Reward: {reward:.2f}")
#         self.total_reward += reward
#         return reward
#
#     def calculate_total_reward(self):
#         return self.total_reward
#
#     def reset_total_reward(self):
#         self.total_reward = 0
#         self.previous_x_position = None
#         self.previous_exit_distance = None
#         self.step_count = 0
#         self.last_health = 100
#         self.consecutive_jumps = 0
#         self.jump_cooldown = 0


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
