from src.ai_agent.agent_state_and_action import GameActions
from src.environment.soldier import Soldier


class RewardAI:
    def __init__(self):
        self.total_reward = 0.0

    def calculate_reward(self, chosen_action: GameActions, player: Soldier, died, reached_exit, enemy_group,
                         bullet_group, has_gained_coin, has_gained_health):
        reward = 0.0

        # Positive rewards
        if reached_exit:
            reward += 1000  # Large reward for reaching the goal
        elif not died:
            reward += 0.5  # Small reward for survival

        # Movement rewards
        if chosen_action == GameActions.MOVE_RIGHT and player.walked_forward():
            reward += 25
        elif chosen_action == GameActions.MOVE_LEFT and player.walked_backward():
            reward += 20
        elif chosen_action == GameActions.STOP:
            reward += 20

        # Combat rewards and penalties
        elif chosen_action == GameActions.SHOOT:
            reward += 50
            hit_enemy = False
            for enemy in enemy_group:
                if player.rect.colliderect(enemy.rect):
                    reward += 50  # Bonus for engaging an enemy
                    hit_enemy = True

            if player.walked_backward():
                for enemy in enemy_group:
                    if player.rect.colliderect(enemy.rect):
                        reward += 70  # Bonus for engaging an enemy

            # Penalty for missed bullets
            if not hit_enemy:
                for bullet in bullet_group:
                    if not any(bullet.rect.colliderect(enemy.rect) for enemy in enemy_group):
                        reward -= 25.0  # Slightly higher penalty for each missed bullet

        elif chosen_action == GameActions.GRENADE:
            reward += 30

        # Jump encouragement
        elif chosen_action == GameActions.JUMP and player.in_air:
            reward += 5.0

        # Coin reward
        if has_gained_coin:
            reward += 5.0

        # Health reward
        if has_gained_health:
            reward += 1.5

        # Survival reward
        if not player.rect.collidelist([e.rect for e in enemy_group]) == -1:
            reward += 40.0  # Small reward for staying alive

        # Negative rewards
        if died:
            reward -= 1000  # Harsh penalty for dying

        # # Penalty for exploit behavior
        # tilex, tiley = world.get_player_tile_position(player.rect.x, player.rect.y)
        # if player.rect.x == 0 or player.rect.x == SCREEN_WIDTH - player.rect.width:
        #     reward -= 1.0  # Penalty for camping at the edges
        #
        # if player.rect.y == 0 or player.rect.y == SCREEN_HEIGHT - player.rect.height:
        #     reward -= 1.0  # Penalty for camping at the edges

        # Accumulate total reward
        self.total_reward += reward

        return reward

    def calculate_total_reward(self):
        return self.total_reward

    def reset_total_reward(self):
        self.total_reward = 0

