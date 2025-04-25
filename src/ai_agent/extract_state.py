from src.settings import *

class ExtractState():
    def extract_state(self, player, enemy_group, exit_group):
        state = []
        # 1. Player's normalized X and Y position
        state.append(player.rect.centerx / SCREEN_WIDTH)
        state.append(player.rect.centery / SCREEN_HEIGHT)

        # 2. Is player in air? (binary)
        state.append(int(player.in_air))

        # 2. Is player taking action? [0: idle, 1: run, 2: jump]
        state.append(player.action)

        # 2. Is player in which direction? [-1: left, 1: right]
        state.append(player.direction)

        # 3. Ammo left with player
        state.append(player.ammo)

        # 4. Grenades leftwith player
        state.append(player.grenades)

        # 5. Health in percentage
        state.append(player.health / player.max_health)

        # 6. Distance to nearest enemy (X and Y, normalized)
        nearest_enemy = self._get_nearest_enemy(player, enemy_group)
        if nearest_enemy:
            dx = (nearest_enemy.rect.centerx - player.rect.centerx) / SCREEN_WIDTH
            dy = (nearest_enemy.rect.centery - player.rect.centery) / SCREEN_HEIGHT
            state.extend([dx, dy])
        else:
            state.extend([1.0, 1.0])  # no enemy nearby

        # 7. Is player near the end of level (binary)
        state.append(int(self._check_exit_nearby(player, exit_group)))

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
