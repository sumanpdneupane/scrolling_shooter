from src.ai_agent.agent_state_and_action import GameActions
from src.environment.soldier import Soldier


# class RewardAI:
#     def __init__(self):
#         self.total_reward = 0
#         # Define reward/penalty values
#         self.MOVE_RIGHT = 5
#         self.MOVE_LEFT = 4
#         self.JUMP = 0.3
#         self.HEALTH_LOSS_PENALTY = -0.5
#         self.HEALTH_GAIN_REWARD = 1.0
#         self.FIRE_BULLETS = 0.2
#         self.AMMO_GAIN_REWARD = 0.3
#         self.GRENADE_GAIN_REWARD = 0.5
#         self.ENEMY_GET_HURT = 0.1
#         self.ENEMY_KILL_REWARD = 2.0
#         self.DEATH_PENALTY = -10.0
#         self.EXIT_REWARD = 10.0
#         self.SHOOT_PENALTY = -0.1
#
#     def calculate_reward(self, chosen_action: GameActions, player: Soldier, enemy_killed, died, reached_exit, shot_fired, prev_health, prev_ammo, prev_grenades):
#         reward = 0
#
#         # Health changes
#         health_diff = player.health - prev_health
#         reward += health_diff * (self.HEALTH_GAIN_REWARD if health_diff > 0 else self.HEALTH_LOSS_PENALTY)
#
#         # Movement
#         if player.jump and chosen_action == GameActions.JUMP:
#             reward += self.JUMP
#         if chosen_action == GameActions.MOVE_RIGHT and (player.walked_forward() or player.direction == 1):
#             reward += self.MOVE_RIGHT
#         if chosen_action == GameActions.MOVE_LEFT and (player.walked_backward() or player.direction == -1):
#             reward += self.MOVE_LEFT
#
#
#         # Ammo/grenade gains
#         if player.ammo > prev_ammo:
#             reward += self.AMMO_GAIN_REWARD
#         elif player.grenades > prev_grenades:
#             reward += self.GRENADE_GAIN_REWARD
#
#         # Enemy killed
#         elif player.bullet_hit_enemy():
#             reward += self.ENEMY_GET_HURT
#         elif enemy_killed:
#             reward += self.ENEMY_KILL_REWARD
#
#         # Terminal states
#         elif died:
#             reward += self.DEATH_PENALTY
#         elif reached_exit:
#             reward += self.EXIT_REWARD
#
#         self.total_reward += reward
#         return reward
#
#     def calculate_total_reward(self):
#         return self.total_reward
#
#     def reset_total_reward(self):
#         self.total_reward = 0


# class RewardAI:
#     def __init__(self):
#         self.total_reward = 0
#
#         # Core Movement
#         self.MOVE_RIGHT = 5
#         self.MOVE_LEFT = -5 #-20.0
#         self.STOP = 2
#         self.JUMP_REWARD = 0.1
#         self.JUMP_REWARD_PENALTY = -1.0
#         self.SMOOTH_LANDING_REWARD = 0.2
#
#         # Combat and Interaction
#         self.ENEMY_HIT_REWARD = 2.0
#         self.ENEMY_KILL_REWARD = 7.0 #50.0
#         self.AMMO_GAIN_REWARD = 0.5
#         self.GRENADE_GAIN_REWARD = 0.5
#         self.SHOOT_PENALTY = -2.0 #-0.05
#         self.THROW_GRENADE = -3.0
#
#         # Health
#         self.HEALTH_GAIN_REWARD = 10.0
#         self.HEALTH_LOSS_PENALTY = -15.0 #-40.0
#
#         # Critical States
#         self.ALIVE = 0.01
#         self.DEATH_PENALTY = -500 #-2000.0 * self.MOVE_RIGHT / 2
#         self.EXIT_REWARD = 500 #2000.0
#
#     def calculate_reward(self, chosen_action: GameActions, player: Soldier, enemy_killed, died, reached_exit,
#                          shot_fired, prev_health, prev_ammo, prev_grenades):
#         reward = 0
#
#         # Health changes
#         health_diff = player.health - prev_health
#         if health_diff < 0:
#             reward += self.HEALTH_LOSS_PENALTY # (self.HEALTH_LOSS_PENALTY * abs(health_diff) / 2.5)
#         elif health_diff > 0:
#             reward += self.HEALTH_GAIN_REWARD #(self.HEALTH_GAIN_REWARD * abs(health_diff) / 5)
#
#         # Movement
#         if chosen_action == GameActions.MOVE_RIGHT and player.walked_forward():
#             reward += self.MOVE_RIGHT
#         elif chosen_action == GameActions.MOVE_LEFT and player.walked_backward():
#             reward += self.MOVE_LEFT
#         elif chosen_action == GameActions.STOP:
#             reward += self.STOP
#
#         # Jumping and landing
#         elif chosen_action == GameActions.JUMP:
#             if player.jump:
#                 reward += self.JUMP_REWARD
#             elif player.in_air:
#                 reward += self.JUMP_REWARD_PENALTY
#
#         # AMMO or GRENADE collection
#         elif chosen_action == GameActions.SHOOT or chosen_action == GameActions.GRENADE:
#             if player.ammo > prev_ammo:
#                 reward += (2 * self.AMMO_GAIN_REWARD * (player.max_ammo - player.ammo))
#             elif player.grenades > prev_grenades:
#                 reward += (2 * self.GRENADE_GAIN_REWARD * (player.max_ammo - player.ammo))
#
#             # Ammo and grenade collection
#             elif player.ammo < prev_ammo:
#                 reward -= (self.AMMO_GAIN_REWARD * (player.max_ammo - player.ammo)) #(4 * self.AMMO_GAIN_REWARD * (player.max_ammo - player.ammo)) ** 1.25
#             elif player.grenades < prev_grenades:
#                 reward -=  (self.GRENADE_GAIN_REWARD * (player.max_ammo - player.ammo)) #(4 * self.GRENADE_GAIN_REWARD * (player.max_ammo - player.ammo)) ** 1.25
#
#             # Combat rewards
#             elif player.bullet_hit_enemy():
#                 reward += self.ENEMY_HIT_REWARD
#             elif enemy_killed:
#                 reward += self.ENEMY_KILL_REWARD
#
#         # Terminal states
#         elif died:
#             reward += self.DEATH_PENALTY
#         elif not died:
#             reward += self.ALIVE
#         elif reached_exit:
#             reward += self.EXIT_REWARD
#
#         # Update total reward
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
        self.total_reward = 0

        # Core Movement
        self.MOVE_RIGHT = 8.0
        self.MOVE_LEFT = -3.0
        self.STOP = 0.5
        self.JUMP_REWARD = 0.2
        self.JUMP_REWARD_PENALTY = -0.5
        self.EDGE_JUMP_REWARD = 5.0
        self.SMOOTH_LANDING_REWARD = 0.2

        # Combat and Interaction
        self.ENEMY_HIT_REWARD = 5.0
        self.ENEMY_KILL_REWARD = 20.0
        self.AMMO_GAIN_REWARD = 2.0
        self.GRENADE_GAIN_REWARD = 2.0
        self.SHOOT_PENALTY = -1.0
        self.THROW_GRENADE = -2.0

        # Health
        self.HEALTH_GAIN_REWARD = 10.0
        self.HEALTH_LOSS_PENALTY = -10.0

        # Critical States
        self.ALIVE = 1.0
        self.DEATH_PENALTY = -1000.0
        self.EXIT_REWARD = 500.0
        self.DEATH_FALL = -1500.0

    def calculate_reward(self, chosen_action: GameActions, player: Soldier, enemy_killed, died, reached_exit,
                         shot_fired, prev_health, prev_ammo, prev_grenades, world):
        reward = 0

        # Health changes
        health_diff = player.health - prev_health
        if health_diff < 0:
            reward += self.HEALTH_LOSS_PENALTY * (abs(health_diff) / player.max_health)
        elif health_diff > 0:
            reward += self.HEALTH_GAIN_REWARD * (abs(health_diff) / player.max_health)

        # Movement
        elif chosen_action == GameActions.MOVE_RIGHT and player.walked_forward():
            reward += self.MOVE_RIGHT
        elif chosen_action == GameActions.MOVE_LEFT and player.walked_backward():
            reward += self.MOVE_LEFT
        elif chosen_action == GameActions.STOP:
            reward += self.STOP

        # Jumping and landing
        elif chosen_action == GameActions.JUMP:
            # if player.jump and player.player_near_edge(world=world):
            #     reward += self.EDGE_JUMP_REWARD  # Reward for jumping near edge
            # el
            if player.jump:
                reward += self.JUMP_REWARD
            elif player.in_air:
                reward += self.JUMP_REWARD_PENALTY

        # Ammo and Grenade Collection
        elif player.ammo > prev_ammo:
            reward += self.AMMO_GAIN_REWARD
        elif player.ammo < prev_ammo:
            reward += self.SHOOT_PENALTY

        if player.grenades > prev_grenades:
            reward += self.GRENADE_GAIN_REWARD
        elif player.grenades < prev_grenades:
            reward += self.THROW_GRENADE

        # Combat rewards
        elif player.bullet_hit_enemy():
            reward += self.ENEMY_HIT_REWARD
        elif enemy_killed:
            reward += self.ENEMY_KILL_REWARD

        # Terminal states (priority to death and exit)
        elif player.fell_or_hit_water():
            reward += self.DEATH_FALL
        elif died:
            reward += self.DEATH_PENALTY
        elif reached_exit:
            reward += self.EXIT_REWARD
        else:
            reward += self.ALIVE

        # Update total reward
        self.total_reward += reward
        return reward

    def calculate_total_reward(self):
        return self.total_reward

    def reset_total_reward(self):
        self.total_reward = 0

