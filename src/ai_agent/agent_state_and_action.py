import enum
import numpy as np

from src.environment.soldier import Soldier
from src.settings import *


class ExtractGameState():
    def extract_state(self, player, world, enemy_group, exit_group):
        state = {}

        # 1. Player's normalized X and Y position
        state["player_pos_x"] = player.rect.centerx / SCREEN_WIDTH
        state["player_pos_y"] = player.rect.centery / SCREEN_HEIGHT

        # 2. Is player in air? (binary)
        state["player_in_air"] = int(player.in_air)

        # 3. Is player taking action? [0: idle, 1: run, 2: jump, 3: death]
        state["player_action"] = player.action

        # 4. Is player moving in which direction? [-1: left, 1: right]
        state["player_direction"] = player.direction

        # 5. Ammo left with player
        state["ammo"] = player.ammo

        # 6. Grenades left with player
        state["grenades"] = player.grenades

        # 7. Health in percentage
        state["health_percentage"] = player.health / player.max_health

        # 8. Distance to nearest enemy (X and Y, normalized)
        nearest_enemy = self._get_nearest_enemy(player, enemy_group)
        if nearest_enemy:
            dx = (nearest_enemy.rect.centerx - player.rect.centerx) / SCREEN_WIDTH
            dy = (nearest_enemy.rect.centery - player.rect.centery) / SCREEN_HEIGHT
            state["nearest_enemy_dx"] = dx
            state["nearest_enemy_dy"] = dy
        else:
            state["nearest_enemy_dx"] = 1.0  # no enemy nearby
            state["nearest_enemy_dy"] = 1.0

        # 2. nearest_enemy_health
        nearest_enemy_health = 100
        min_dist = float('inf')
        for enemy in enemy_group:
            dist = abs(player.rect.centerx - enemy.rect.centerx)
            if dist < min_dist:
                min_dist = dist
                nearest_enemy_health = enemy.health
        state["nearest_enemy_health"] = nearest_enemy_health / 100 # normalize 0-1

        # 3. number_of_enemies_nearby
        nearby_enemies = sum(1 for enemy in enemy_group if player.rect.colliderect(enemy.vision))
        state["nearby_enemies"] = nearby_enemies / 10  # assume max 10 enemies near (normalize)

        # 9. Is player near the exit (binary)
        state["near_exit"] = int(self._check_exit_nearby(player, exit_group))

        # 10. Is player on ground? (binary)
        state["on_ground"] = int(self._is_on_ground(player))

        # 4. distance_to_ground_below
        ground_distance = SCREEN_HEIGHT
        player_bottom = player.rect.bottom
        player_centerx = player.rect.centerx
        for tile_img, tile_rect in world.obstacle_list:
            if tile_rect.collidepoint(player_centerx, tile_rect.top) and tile_rect.top > player_bottom:
                dist = tile_rect.top - player_bottom
                ground_distance = min(ground_distance, dist)
        state["ground_distance"] = min(ground_distance, 500) / 500  # normalize (assuming 500px max)

        # 11. Is player in water? (binary)
        state["in_water"] = int(self._is_in_water(player, water_group))

        # 12. Is player in space? (binary)
        state["in_space"] = int(self._is_in_space(player))

        # 6. current_platform_type
        platform_type = 0  # 0: normal ground, 1: water, 2: ice (future if you add ice tiles)
        if pygame.sprite.spritecollide(player, water_group, False):
            platform_type = 1
        state["platform_type"] = platform_type

        # 7. available_pickups_nearby
        pickup_nearby = any(
            item_box.rect.colliderect(pygame.Rect(player.rect.centerx - 200, player.rect.centery - 200, 400, 400))
            for item_box in item_box_group
        )
        state["pickup_nearby"] = pickup_nearby

        # 5. is_under_attack
        # if not hasattr(player, 'prev_health'):
        #     player.prev_health = player.health
        #     player.hurt_timer = 0
        # if player.health < player.prev_health:
        #     player.hurt_timer = 20  # stay hurt for 20 frames
        # is_under_attack = int(player.hurt_timer > 0)
        # inputs.append(is_under_attack)
        # player.hurt_timer = max(0, player.hurt_timer - 1)
        # player.prev_health = player.health

        # 8. time_left (optional: if you add timer)
        # Let's assume you have a 'game_timer' somewhere
        # if 'game_timer' in globals():
        #     inputs.append(game_timer / total_time)  # normalized 0-1
        # else:
        #     inputs.append(1.0)  # full time left if no timer

        return np.array(list(state.values()), dtype=np.float32)

    def _get_nearest_enemy(self, player, enemy_group):
        min_dist = float('inf')
        nearest = None
        for enemy in enemy_group:
            dist = abs(enemy.rect.centerx - player.rect.centerx)
            if dist < min_dist:
                min_dist = dist
                nearest = enemy
        return nearest

    def _check_exit_nearby(self, player, exit_group):
        for exit in exit_group:
            if abs(exit.rect.centerx - player.rect.centerx) < 150:
                return True
        return False

    # Helper method to check if the player is on the ground
    def _is_on_ground(self, player):
        return player.rect.bottom >= GROUND_THRESHOLD  # Adjust threshold as needed

    # Helper method to check if the player is in water
    def _is_in_water(self, player, water_group):
        if pygame.sprite.spritecollide(player, water_group, False):
            return True
        return False

    # Helper method to check if the player is in space (high altitude or no terrain nearby)
    def _is_in_space(self, player):
        # Check if the player is off the screen (out of the screen bounds)
        # if player.rect.right < 0 or player.rect.left > SCREEN_WIDTH or player.rect.bottom < 0 or player.rect.top > SCREEN_HEIGHT:
        #     # Player is out of screen bounds, consider them in space zone
        #     return True
        # return False
        in_space = int(player.rect.top > SCREEN_HEIGHT or player.rect.left < 0 or player.rect.right > SCREEN_WIDTH)
        return in_space


class GameActions(enum.IntEnum):
    No_action = 0
    MoveLeft = 1
    MoveRight = 2
    Jump = 3
    Shoot = 4
    Grenade = 5
