from src.ai_agent.agent_state_and_action import GameActions
from src.environment.soldier import Soldier
from src.environment.world import TILE_SIZE_W


# class RewardAI:
#     def __init__(self):
#         self.total_reward = 0.0
#
#     def calculate_reward(self, chosen_action: GameActions, player: Soldier,died, reached_exit, world, enemy_group, allow_jump):
#
#         # Initialize reward
#         reward = 0
#
#         # if player.direction == -1:
#         #     reward -= 1
#
#         # if GameActions.MOVE_RIGHT and player.has_moved_one_tile_directional() == 1:
#         #     reward += 20.0
#         # elif chosen_action == GameActions.MOVE_LEFT and player.has_moved_one_tile_directional() == -1:
#         #     reward -= 0.5
#         # elif chosen_action == GameActions.STOP and player.has_moved_one_tile_directional(TILE_SIZE) == 0:
#         #     reward += 0.05  # Slight reward for staying still (you can adjust this)
#
#         exit_position =  (6100/player.distance_to_exit()) ** 1.8
#
#         print(f"allow_jump: {allow_jump}")
#
#         if chosen_action == GameActions.MOVE_RIGHT and player.walked_forward():
#             if not allow_jump:
#                 reward -= 5.0 # Apply penalty for trying to move when jumping isn't allowed
#             else:
#                 reward += 25.0 * exit_position  # Reward if moving right is valid
#         elif chosen_action == GameActions.MOVE_LEFT and player.walked_backward():
#             reward -= 5.0
#
#         if chosen_action == GameActions.JUMP and allow_jump:
#             if allow_jump:
#                 if player.in_air:  # Only reward if the jump was possible
#                     # Check if the jump covers one tile distance
#                     # pre_jump_x = player.prev_x # Assuming this method exists
#                     # post_jump_x = player.rect.x
#                     # distance_travelled = abs(post_jump_x - pre_jump_x)
#                     print(f"----------------------------distance_travelled: {player.walked_forward()}")
#
#                     # If the player jumps over one tile distance (you can define TILE_SIZE)
#                     # if distance_travelled >= 5:  # TILE_SIZE_W is the width of one tile
#                     if player.walked_forward():
#                         reward += 25.0  # High reward for jumping one tile
#                     else:
#                         reward += 5.0  # Reward for regular jump
#                 else:
#                     reward -= 5.0  # Penalty if they attempt to jump but it's not possible
#             else:
#                 reward -= 10.0  # Penalize for trying to jump when jumping is not allowed
#
#         if chosen_action == GameActions.STOP:
#             reward += 5.0
#
#         # elif chosen_action == GameActions.SHOOT:
#         #     if player.get_nearest_enemy(enemy_group):
#         #         if player.bullet_hit_enemy():  # You can track this in update logic
#         #             reward += 0.5
#         #         else:
#         #             reward -= 0.2
#         #     else:
#         #         reward -= 1.0  # Shooting blindly is discouraged
#
#         # elif chosen_action == GameActions.GRENADE:
#         #     if player.get_nearest_enemy(enemy_group):
#         #         if player.hit_enemy_with_grenade:
#         #             reward += 2.0
#         #         else:
#         #             reward -= 0.5
#         #     else:
#         #         reward -= 1.0  # Grenade wasted
#
#         # Extra: Minor penalties or bonuses for special conditions
#         # if player.player_near_edge(world):
#         #     reward -= 0.2
#         # if player.alive:
#         #     reward += 0.5
#         # if player.reached_exit():
#         #     reward += 50.0
#
#         if player.alive:
#             reward += 0.1
#
#         # If the player is too close to an edge
#         if player.player_near_edge(world):
#             reward -= 0.5  # Penalty for standing too close to the edge
#
#         if not player.alive or player.fell_or_hit_water():
#             reward -= 2000
#
#         if reached_exit:
#             reward += 2000
#
#
#         self.total_reward += reward
#         return reward
#
#     def calculate_total_reward(self):
#         return self.total_reward
#
#     def reset_total_reward(self):
#         self.total_reward = 0


class RewardAI:
    def __init__(self):
        self.total_reward = 0.0

    def calculate_reward(self, chosen_action: GameActions, player: Soldier, died, reached_exit, world, enemy_group, allow_jump):
        reward = 0.0

        # Positive rewards
        if reached_exit:
            reward += 100  # Large reward for reaching the goal
        elif not died:
            # Reward for moving right (progress)
            reward += 0.1

            # Encourage jumps if allowed
            if chosen_action == GameActions.JUMP and allow_jump:
                reward += 0.5

            # Reward for attacking enemies
            for enemy in enemy_group:
                if player.rect.colliderect(enemy.rect):
                    reward += 10  # Bonus for engaging an enemy

            # Reward for avoiding enemies and hazards
            if not player.rect.collidelist([e.rect for e in enemy_group]) == -1:
                reward += 1.0  # Small reward for staying alive

        # Negative rewards
        if died:
            reward -= 100  # Harsh penalty for dying
        if player.direction == -1:
            reward -= 0.1  # Penalize moving left if the goal is to move right

        # Apply scaling to the reward
        scaled_reward = reward / 10  # Scale down to stabilize training

        # Accumulate total reward
        self.total_reward += scaled_reward

        return scaled_reward

    def calculate_total_reward(self):
        return self.total_reward

    def reset_total_reward(self):
        self.total_reward = 0

