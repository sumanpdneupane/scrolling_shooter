import enum
import numpy as np
from src.settings import *

class GameActions(enum.IntEnum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    JUMP = 2
    SHOOT = 3
    GRENADE = 4
    STOP = 5


class ExtractGameState:
    def extract_state(self, player, world, enemy_group, exit_group):
        # Convert positions to tile coordinates
        player_tile_x = player.rect.centerx // TILE_SIZE
        player_tile_y = player.rect.centery // TILE_SIZE

        # Initialize state dictionary with normalized values
        state = {
            # Core player position and state
            "player_tile_x": player_tile_x / (SCREEN_WIDTH // TILE_SIZE),  # Normalized x position
            "player_tile_y": player_tile_y / (SCREEN_HEIGHT // TILE_SIZE),  # Normalized y position
            "player_direction": player.direction,  # -1 or 1
            "in_air": float(player.in_air),
            "on_ground": float(self._is_on_ground(player)),

            # Player stats
            "health": player.health / player.max_health,  # Normalized health
            "ammo": min(player.ammo / 20.0, 1.0),  # Normalized with cap
            "grenades": min(player.grenades / 5.0, 1.0),  # Normalized with cap

            # Important environmental features - pre-calculated
            "path_clear": float(self._is_path_clear(player, world)),
            "water_ahead": float(self._check_water_ahead(player)),
            "water_below": float(self._check_water_below(player)),
            "space_ahead": float(self._check_space_ahead(player, world)),
            "wall_ahead": float(not self._is_path_clear(player, world)),
            "on_edge": float(self._is_on_edge(player, world)),

            # Local obstacle detection (directional)
            "obstacle_right": float(self._check_tile_side_hit(player, world, 1)),
            "obstacle_left": float(self._check_tile_side_hit(player, world, -1)),
            "ground_below": float(not self._check_space_below(player, world)),

            # Distance measurements
            "ground_distance": self._calculate_ground_distance(player, world) / 10.0,  # Normalized
        }

        # Add exit position and distances if available
        if exit_group:
            exit = exit_group.sprites()[0]
            exit_tile_x = exit.rect.centerx // TILE_SIZE
            exit_tile_y = exit.rect.centery // TILE_SIZE

            # Calculate distance to exit (normalized by screen size)
            dx = exit_tile_x - player_tile_x
            dy = exit_tile_y - player_tile_y

            # Normalize direction to exit
            max_distance = max(SCREEN_WIDTH, SCREEN_HEIGHT) // TILE_SIZE
            norm_dx = dx / max_distance
            norm_dy = dy / max_distance

            # Calculate Euclidean distance (normalized)
            distance = ((dx ** 2) + (dy ** 2)) ** 0.5
            norm_distance = min(distance / max_distance, 1.0)

            state.update({
                "exit_tile_x": norm_dx,
                "exit_tile_y": norm_dy,
                "exit_distance": norm_distance,
                "near_exit": float(distance < 7),
            })
        else:
            # Default values if no exit is available
            state.update({
                "exit_tile_x": 0.0,
                "exit_tile_y": 0.0,
                "exit_distance": 1.0,
                "near_exit": 0.0,
            })

        # Enemy information (nearest enemy)
        nearest_enemy = self._get_nearest_enemy(player, enemy_group)
        if nearest_enemy:
            enemy_tile_x = nearest_enemy.rect.centerx // TILE_SIZE
            enemy_tile_y = nearest_enemy.rect.centery // TILE_SIZE

            # Calculate normalized vector to enemy
            dx = enemy_tile_x - player_tile_x
            dy = enemy_tile_y - player_tile_y
            max_distance = max(SCREEN_WIDTH, SCREEN_HEIGHT) // TILE_SIZE

            state.update({
                "enemy_direction_x": dx / max_distance,
                "enemy_direction_y": dy / max_distance,
                "enemy_distance": min(((dx ** 2) + (dy ** 2)) ** 0.5 / max_distance, 1.0),
                "enemy_health": nearest_enemy.health / 100.0,
            })
        else:
            # Default values if no enemies
            state.update({
                "enemy_direction_x": 0.0,
                "enemy_direction_y": 0.0,
                "enemy_distance": 1.0,
                "enemy_health": 0.0,
            })

        # Convert to numpy array for the neural network
        state_array = np.array(list(state.values()), dtype=np.float32)

        return state, state_array

    # Helper methods
    def _is_on_ground(self, player):
        return not player.in_air

    def _get_nearest_enemy(self, player, enemy_group):
        if not enemy_group:
            return None

        return min(
            enemy_group,
            key=lambda e: ((e.rect.centerx - player.rect.centerx) ** 2 +
                           (e.rect.centery - player.rect.centery) ** 2) ** 0.5,
            default=None
        )

    def _calculate_ground_distance(self, player, world):
        """Calculate distance to ground below player"""
        player_bottom = player.rect.bottom
        for y_offset in range(1, 11):  # Check up to 10 tiles down
            check_y = player_bottom + (y_offset * TILE_SIZE)
            if any(tile_rect.collidepoint(player.rect.centerx, check_y)
                   for tile_img, tile_rect in world.obstacle_list):
                return y_offset
        return 10  # Maximum return if no ground found

    def _is_path_clear(self, player, world):
        """Check if the path ahead is clear of obstacles"""
        # Check 2 tiles ahead in movement direction
        look_ahead = player.rect.centerx + (2 * TILE_SIZE * player.direction)
        return not any(
            tile_rect.collidepoint(look_ahead, player.rect.centery)
            for tile_img, tile_rect in world.obstacle_list
        )

    def _check_water_ahead(self, player):
        """Check for water in the next 2 tiles ahead"""
        future_x = player.rect.centerx + (2 * TILE_SIZE * player.direction)
        future_y = player.rect.bottom - (TILE_SIZE // 2)
        return any(
            water.rect.collidepoint(future_x, future_y)
            for water in water_group
        )

    def _check_water_below(self, player):
        """Check for water directly below player"""
        check_x = player.rect.centerx
        check_y = player.rect.bottom + (TILE_SIZE // 2)
        return any(
            water.rect.collidepoint(check_x, check_y)
            for water in water_group
        )

    def _check_space_ahead(self, player, world):
        """Check if there's a gap in the ground ahead"""
        # Check for ground 2 tiles ahead and 1 tile below
        check_x = player.rect.centerx + (2 * TILE_SIZE * player.direction)
        check_y = player.rect.bottom + (TILE_SIZE // 2)
        return not any(
            tile_rect.collidepoint(check_x, check_y)
            for tile_img, tile_rect in world.obstacle_list
        )

    def _check_space_below(self, player, world):
        """Check if there's ground directly below player"""
        check_x = player.rect.centerx
        check_y = player.rect.bottom + (TILE_SIZE // 2)
        return not any(
            tile_rect.collidepoint(check_x, check_y)
            for tile_img, tile_rect in world.obstacle_list
        )

    def _check_tile_side_hit(self, player, world, direction):
        """Check if there's an obstacle to the left/right of player"""
        check_x = player.rect.centerx + (direction * TILE_SIZE)
        check_y = player.rect.centery
        return any(
            tile_rect.collidepoint(check_x, check_y)
            for tile_img, tile_rect in world.obstacle_list
        )

    def _is_on_edge(self, player, world):
        """Check if player is standing at the edge of a platform"""
        if player.in_air:
            return False

        # Check if there's ground directly below
        below_solid = any(
            tile_rect.collidepoint(player.rect.centerx, player.rect.bottom + (TILE_SIZE // 2))
            for tile_img, tile_rect in world.obstacle_list
        )

        # Check if there's no ground ahead
        ahead_empty = not any(
            tile_rect.collidepoint(
                player.rect.centerx + (TILE_SIZE * player.direction),
                player.rect.bottom + (TILE_SIZE // 2)
            )
            for tile_img, tile_rect in world.obstacle_list
        )

        return below_solid and ahead_empty