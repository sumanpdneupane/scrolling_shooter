from src.settings import *

class ExtractState():
    def extract_state(self, player, enemy_group, exit_group):
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

        # 9. Is player near the exit (binary)
        state["near_exit"] = int(self._check_exit_nearby(player, exit_group))

        return state

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
