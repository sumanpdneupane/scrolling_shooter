import numpy as np
from src.ai_agent.agent_state_and_action import GameActions

class RewardAI:
    def __init__(self):
        self.total_reward = 0
        self.previous_exit_distance = None

    def calculate_reward(self, state_dict, state, action, died):
        reward = 0

        reward += self._movement_reward(state_dict, action)
        reward += self._exit_reward(state_dict)
        reward += 0.005  # small exploration bonus
        if died:
            reward = -1000  # large negative reward for death

        # reward = np.clip(reward, -1.0, 1.0)

        print(f"Action: {action}, Reward: {reward:.4f}, "
              f"GroundDist: {state_dict['ground_distance']:.2f}, InWater: {state_dict['in_water']}, "
              f"PathClear: {state_dict['path_clear']}, WaterAhead: {state_dict['water_ahead']}, "
              f"WaterBelow: {state_dict['water_below']}, SpaceAhead: {state_dict['space_ahead']}, "
              f"LeftHit: {state_dict['tile_left_hit']}, RightHit: {state_dict['tile_right_hit']}, "
              f"Died: {died}")

        self.total_reward += reward
        return reward

    def _movement_reward(self, state_dict, action):
        reward = 0

        on_ground = bool(state_dict["on_ground"])
        in_water = bool(state_dict["in_water"])
        ground_distance = state_dict["ground_distance"]
        path_clear = bool(state_dict["path_clear"])
        water_ahead = bool(state_dict["water_ahead"])

        should_jump = in_water or ground_distance > 0.4 or not path_clear

        # Walk when on ground and path is clear
        if on_ground and not should_jump:
            if action in [GameActions.MOVE_LEFT, GameActions.MOVE_RIGHT]:
                reward += 0.5  # encourage walking
            elif action == GameActions.JUMP:
                reward -= 0.4  # penalize unnecessary jump

        # Jump if needed (water, gaps, or wall)
        if should_jump:
            if action == GameActions.JUMP:
                reward += 0.6  # correct behavior
            elif action in [GameActions.MOVE_LEFT, GameActions.MOVE_RIGHT]:
                reward -= 0.3  # discourage walking into obstacle/gap

        return reward

    # def __init__(self):
    #     self.total_reward = 0
    #     self.previous_exit_distance = None
    #     self.last_exit_direction = 1  # 1=right, -1=left
    #
    # def calculate_reward(self, state_dict, state, action, died):
    #
    #     # Update exit direction tracking
    #     exit_dx = state_dict["exit_distance_x"]
    #     if exit_dx != 0:
    #         self.last_exit_direction = 1 if exit_dx > 0 else -1
    #
    #     reward = 0
    #     reward += self._movement_reward(state_dict, action)
    #     reward += self._exit_reward(state_dict)
    #     reward += 0.005  # exploration bonus
    #
    #     if died:
    #         reward = -1000  # large negative reward for death
    #
    #     reward = np.clip(reward, -1.0, 1.0)
    #
    #     print(f"Action: {action}, Reward: {reward:.4f}, "
    #           f"GroundDist: {state_dict['ground_distance']:.2f}, InWater: {state_dict['in_water']}, "
    #           f"PathClear: {state_dict['path_clear']}, WaterAhead: {state_dict['water_ahead']}, "
    #           f"WaterBelow: {state_dict['water_below']}, SpaceAhead: {state_dict['space_ahead']}, "
    #           f"LeftHit: {state_dict['tile_left_hit']}, RightHit: {state_dict['tile_right_hit']}, "
    #           f"Died: {died}")
    #
    #     self.total_reward += reward
    #     return reward
    #
    # def _movement_reward(self, state_dict, action):
    #     reward = 0
    #     on_ground = state_dict["on_ground"]
    #     in_water = state_dict["in_water"]
    #     ground_distance = state_dict["ground_distance"]
    #     path_clear = state_dict["path_clear"]
    #     space_ahead = state_dict["space_ahead"]
    #     water_ahead = state_dict["water_ahead"]
    #
    #     # Determine required movement direction based on exit position
    #     target_direction = self.last_exit_direction
    #     moving_forward = (action == GameActions.MOVE_RIGHT if target_direction == 1
    #                       else action == GameActions.MOVE_LEFT)
    #
    #
    #
    #     # Jump required for obstacles/gaps/water
    #     should_jump = (
    #             (not path_clear and not on_ground) or  # Wall jumping
    #             (space_ahead and ground_distance > 1.0) or  # Gap
    #             water_ahead or
    #             in_water
    #     )
    #
    #     if should_jump:
    #         if action == GameActions.JUMP:
    #             reward += 0.6
    #             # Encourage moving in target direction while jumping
    #             if moving_forward:
    #                 reward += 0.3
    #         elif moving_forward:
    #             # Don't penalize forward movement during needed jumps
    #             reward -= 0.1
    #         else:
    #             reward -= 0.2
    #     else:
    #         if moving_forward:
    #             reward += 0.7  # Strong reward for forward progress
    #         elif action in [GameActions.MOVE_LEFT, GameActions.MOVE_RIGHT]:
    #             reward += 0.3  # Some reward for lateral movement
    #
    #         if action == GameActions.JUMP and not (space_ahead or water_ahead):
    #             reward -= 0.5  # Penalize unnecessary jumping
    #
    #     return reward

    def _exit_reward(self, state_dict):
        exit_reward = 0
        # current_distance = np.sqrt(
        #     state_dict["exit_distance_x"] ** 2 + state_dict["exit_distance_y"] ** 2
        # )
        current_distance = np.sqrt(
            state_dict["exit_tile_x"] ** 2 + state_dict["exit_tile_y"] ** 2
        )

        if self.previous_exit_distance is not None:
            delta = self.previous_exit_distance - current_distance
            exit_reward += np.clip(delta * 2.0, -0.5, 1.0)

        self.previous_exit_distance = current_distance

        if state_dict["near_exit"]:
            exit_reward += 5.0

        return exit_reward

    def calculate_total_reward(self):
        return self.total_reward

    def reset_total_reward(self):
        self.total_reward = 0
        self.previous_exit_distance = None
