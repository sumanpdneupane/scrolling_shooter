import enum
import numpy as np
from src.settings import *


# class ExtractGameState:
#     def extract_state(self, player, world, enemy_group, exit_group):
#         state = {
#             # Player status
#             "player_pos_x": player.rect.centerx / SCREEN_WIDTH,
#             "player_pos_y": player.rect.centery / SCREEN_HEIGHT,
#             "in_air": int(player.in_air),
#             "action": player.action,
#             "direction": player.direction,
#             "ammo": player.ammo,
#             "grenades": player.grenades,
#             "health": player.health / player.max_health,
#
#             # Enemy information
#             "nearest_enemy_dx": 0.0,
#             "nearest_enemy_dy": 0.0,
#             "nearest_enemy_health": 0.0,
#             "nearby_enemies": 0.0,
#
#             # Environment state
#             "exit_distance_x": 0.0,
#             "exit_distance_y": 0.0,
#             "near_exit": 0,
#             "on_ground": int(self._is_on_ground(player)),
#             "ground_distance": 0.0,
#             "in_water": 0,
#             "platform_type": 0,
#             "pickup_nearby": 0
#         }
#
#         # Enemy calculations
#         nearest_enemy = self._get_nearest_enemy(player, enemy_group)
#         if nearest_enemy:
#             state.update({
#                 "nearest_enemy_dx": (nearest_enemy.rect.centerx - player.rect.centerx) / SCREEN_WIDTH,
#                 "nearest_enemy_dy": (nearest_enemy.rect.centery - player.rect.centery) / SCREEN_HEIGHT,
#                 "nearest_enemy_health": nearest_enemy.health / 100
#             })
#
#         state["nearby_enemies"] = sum(
#             1 for enemy in enemy_group
#             if player.rect.colliderect(enemy.vision)
#         ) / 10
#
#         # Exit calculations
#         if exit_group:
#             exit = exit_group.sprites()[0]
#             state.update({
#                 "exit_distance_x": (exit.rect.centerx - player.rect.centerx) / SCREEN_WIDTH,
#                 "exit_distance_y": (exit.rect.centery - player.rect.centery) / SCREEN_HEIGHT,
#                 "near_exit": int(abs(exit.rect.centerx - player.rect.centerx) < 150)
#             })
#
        # # Environment interactions
        # state.update({
        #     "ground_distance": self._calculate_ground_distance(player, world),
        #     "in_water": True if pygame.sprite.spritecollide(player, water_group, False) else False,
        #     "platform_type": 1 if pygame.sprite.spritecollide(player, water_group, False) else 0,
        #     "pickup_nearby": any(
        #         item_box.rect.colliderect(pygame.Rect(
        #             player.rect.centerx - 200,
        #             player.rect.centery - 200,
        #             400, 400
        #         ))
        #         for item_box in item_box_group
        #     )
        # })
#
#         # Add these state features
#         state.update({
#             "path_clear": int(self._is_path_clear(player, world)),
#             "moving_forward": int(player.direction == 1),
#             "space_ahead": self._check_space_ahead(player, world)
#         })
#
#         # Update water conditions
#         state.update({
#             "water_ahead": self._check_water_ahead(player, world),
#             "water_below": self._check_water_below(player, world)  # New check for water below
#         })
#
#         # 1 if the player’s side is touching a tile. 0 otherwise.
#         state.update({
#             "tile_left_hit": int(self._check_tile_side_hit(player, world, direction=-1)),
#             "tile_right_hit": int(self._check_tile_side_hit(player, world, direction=1))
#         })
#
#         return state, np.array(list(state.values()), dtype=np.float32)
#
#     # Helper methods
#     def _check_tile_side_hit(self, player, world, direction):
#         """
#         Check if the player is touching a tile on the left (direction = -1) or right (direction = 1).
#         """
#         offset = TILE_SIZE * 2
#         check_x = player.rect.centerx + direction * offset
#         check_y_top = player.rect.top + 5
#         check_y_bottom = player.rect.bottom - 5
#
#         return any(
#             tile_rect.collidepoint(check_x, y)
#             for y in range(check_y_top, check_y_bottom)
#             for tile_img, tile_rect in world.obstacle_list
#         )
#
    # def _get_nearest_enemy(self, player, enemy_group):
    #     return min(
    #         enemy_group,
    #         key=lambda e: abs(e.rect.centerx - player.rect.centerx),
    #         default=None
    #     )
#
#     def _calculate_ground_distance(self, player, world):
#         player_bottom = player.rect.bottom
#         return min(
#             (tile_rect.top - player_bottom
#              for tile_img, tile_rect in world.obstacle_list
#              if tile_rect.collidepoint(player.rect.centerx, tile_rect.top)
#              and tile_rect.top > player_bottom),
#             default=SCREEN_HEIGHT
#         ) / 500
#
    # def _is_on_ground(self, player):
    #     return player.rect.bottom >= GROUND_THRESHOLD
#
#     def _is_path_clear(self, player, world):
#         # Check 200 pixels ahead in movement direction
#         look_ahead = player.rect.centerx + 200 * player.direction
#
#         return not any(tile_rect.collidepoint(look_ahead, player.rect.centery)
#                        for tile_img, tile_rect in world.obstacle_list)
#
#     def _check_water_ahead(self, player, world):
#         # Check for water in the next 300 pixels
#         tile_width = TILE_SIZE * 2
#         future_x = player.rect.centerx + tile_width * player.direction
#         future_y = player.rect.centery + tile_width
#         return any(water.rect.collidepoint(future_x, future_y)
#                    for water in water_group)
#
#
    # def _check_water_below(self, player, world):
    #     # Check for water 2 tile ahead of the player in the direction they're moving
    #     # Calculate the future position 1 tile ahead based on the player's direction
    #     tile_width = TILE_SIZE * 1
    #     future_x = player.rect.centerx + player.direction * tile_width
    #     future_y = player.rect.bottom  # Check slightly below the player's feet
    #
    #     # Check for water at the calculated future position
    #     return any(water.rect.collidepoint(future_x, future_y) for water in water_group)
#
#     def _check_space_ahead(self, player, world):
#         tile_width = TILE_SIZE
#         tile_height = TILE_SIZE
#         future_y = player.rect.bottom + 5  # Just below the player's feet
#
#         # Check at 1 and 2 tiles ahead
#         for step in [1, 2]:
#             future_x = player.rect.centerx + player.direction * tile_width * step
#
#             if any(tile_rect.collidepoint(future_x, future_y)
#                    for tile_img, tile_rect in world.obstacle_list):
#                 return False  # Ground exists in at least one step ahead
#
#         return True  # No ground in next two tiles: it's a hole/space ahead


class ExtractGameState:
    def extract_state(self, player, world, enemy_group, exit_group):
        # Convert all positions to tile coordinates
        player_tile_x = player.rect.centerx // TILE_SIZE
        player_tile_y = player.rect.centery // TILE_SIZE

        state = {
            # Tile-based position features
            "player_tile_x": player_tile_x,
            "player_tile_y": player_tile_y,
            "direction": player.direction,
            "in_air": int(player.in_air),
            "on_ground": int(self._is_on_ground(player)),

            # Inventory/health (normalized)
            "health": player.health / player.max_health,
            "ammo": player.ammo / 10.0,
            "grenades": player.grenades / 5.0,

            # Enemy information
            "nearest_enemy_tile_x": 0,
            "nearest_enemy_tile_y": 0,
            "nearest_enemy_health": 0.0,
            "nearby_enemies": 0.0,

            # Environment state
            "exit_tile_x": 0,
            "exit_tile_y": 0,
            "tiles_to_exit_x": 0,
            "tiles_to_exit_y": 0,
            "near_exit": 0,
            "ground_tiles_below": 0.0,
            "water_ahead": 0,
            "water_below": 0,
            "in_water": 0,
            "ledge_ahead": 0,
            "space_ahead": 0,
            "path_clear": 0,
            "tile_left_hit": 0,
            "tile_right_hit": 0
        }

        # Enemy calculations
        nearest_enemy = self._get_nearest_enemy(player, enemy_group)
        if nearest_enemy:
            state.update({
                "nearest_enemy_tile_x": nearest_enemy.rect.centerx // TILE_SIZE,
                "nearest_enemy_tile_y": nearest_enemy.rect.centery // TILE_SIZE,
                "nearest_enemy_health": nearest_enemy.health / 100.0
            })

        # Count enemies within 4 tiles radius
        state["nearby_enemies"] = sum(
            1 for enemy in enemy_group
            if abs((enemy.rect.centerx // TILE_SIZE) - player_tile_x) <= 4
        ) / 5.0  # Normalize to 0-2 range

        # Exit calculations
        if exit_group:
            exit = exit_group.sprites()[0]
            exit_tile_x = exit.rect.centerx // TILE_SIZE
            exit_tile_y = exit.rect.centery // TILE_SIZE
            state.update({
                "exit_tile_x": exit_tile_x,
                "exit_tile_y": exit_tile_y,
                "tiles_to_exit_x": abs(exit_tile_x - player_tile_x),
                "tiles_to_exit_y": abs(exit_tile_y - player_tile_y),
                "near_exit": int(abs(exit.rect.centerx // TILE_SIZE - player.rect.centerx // TILE_SIZE) < 7)
            })

        # Environment interactions
        state.update({
            "ground_distance": self._calculate_ground_distance(player, world),
            "water_ahead": int(self._check_water_ahead(player)),
            "water_below": int(self._check_water_ahead(player)),
            "ledge_ahead": int(self._check_ledge_ahead(player, world)),
            "space_ahead": int(self._check_space_ahead(player, world)),
            "path_clear": int(self._is_path_clear(player, world)),
            "tile_left_hit": int(self._check_tile_side_hit(player, world, -1)),
            "tile_right_hit": int(self._check_tile_side_hit(player, world, 1)),
            "in_water": 1 if pygame.sprite.spritecollide(player, water_group, False) else 0,
            "platform_type": 1 if pygame.sprite.spritecollide(player, water_group, False) else 0,
            "pickup_nearby": any(
                item_box.rect.colliderect(pygame.Rect(
                    player.rect.centerx - 200,
                    player.rect.centery - 200,
                    400, 400
                ))
                for item_box in item_box_group
            )
        })

        return state, np.array(list(state.values()), dtype=np.float32)

    # Tile-based helper methods
    def _is_on_ground(self, player):
        return player.rect.bottom // TILE_SIZE >= GROUND_THRESHOLD
    def _get_nearest_enemy(self, player, enemy_group):
        """Find nearest enemy based on tile distance in x-axis"""
        player_tile_x = player.rect.centerx // TILE_SIZE

        return min(
            enemy_group,
            key=lambda e: abs((e.rect.centerx // TILE_SIZE) - player_tile_x),
            default=None
        )

    def _calculate_ground_distance(self, player, world):
        player_bottom = player.rect.bottom
        min_distance = min(
            (tile_rect.top - player_bottom
             for tile_img, tile_rect in world.obstacle_list
             if tile_rect.centerx == player.rect.centerx
             and tile_rect.top > player_bottom),
            default=SCREEN_HEIGHT - player_bottom
        )
        return min_distance // TILE_SIZE  # Return tile count

    def _check_water_ahead(self, player):
        look_ahead_x = player.rect.centerx + (TILE_SIZE * 2 * player.direction)
        look_ahead_y = player.rect.centery
        return any(
            water.rect.collidepoint(look_ahead_x, look_ahead_y)
            for water in water_group
        )

    def _check_water_below(self, player):
        """Check for water in the tile directly below and one tile ahead"""
        # Get player's current tile position
        player_tile_x = player.rect.centerx // TILE_SIZE
        player_tile_y = player.rect.centery // TILE_SIZE

        # Calculate tile position to check (1 tile ahead in movement direction)
        check_tile_x = player_tile_x + player.direction
        check_tile_y = (player.rect.bottom // TILE_SIZE)  # Tile directly below feet

        # Check if any water exists at this tile position
        return any(
            (water.rect.centerx // TILE_SIZE == check_tile_x) and
            (water.rect.centery // TILE_SIZE == check_tile_y)
            for water in water_group
        )

    def _check_ledge_ahead(self, player, world):
        check_x = player.rect.centerx + (TILE_SIZE * player.direction)
        check_y = player.rect.bottom + TILE_SIZE
        return not any(
            tile_rect.collidepoint(check_x, check_y)
            for tile_img, tile_rect in world.obstacle_list
        )

    def _is_path_clear(self, player, world):
        """Check if path is clear 4 tiles ahead in movement direction"""
        player_tile_x = player.rect.centerx // TILE_SIZE
        look_ahead_tiles = player_tile_x + (4 * player.direction)  # 4 tiles ahead

        # Check for obstacles at same Y level within tile range
        return not any(
            (tile_rect.centerx // TILE_SIZE == look_ahead_tiles) and
            (abs(tile_rect.centery // TILE_SIZE - (player.rect.centery // TILE_SIZE)) <= 1)
            for tile_img, tile_rect in world.obstacle_list
        )

    def _check_space_ahead(self, player, world):
        """Check for holes 1-2 tiles ahead and below"""
        player_tile_x = player.rect.centerx // TILE_SIZE
        player_tile_y_bottom = (player.rect.bottom) // TILE_SIZE

        for steps in [1, 2]:
            check_tile_x = player_tile_x + (player.direction * steps)
            # Check tile directly below at stepped position
            if any(
                    (tile_rect.centerx // TILE_SIZE == check_tile_x) and
                    (tile_rect.centery // TILE_SIZE == player_tile_y_bottom)
                    for tile_img, tile_rect in world.obstacle_list
            ):
                return False  # Ground exists at this step

        return True  # No ground found at either step

    def _check_tile_side_hit(self, player, world, direction):
        check_x = player.rect.centerx + (direction * TILE_SIZE)
        return any(
            tile_rect.collidepoint(check_x, player.rect.centery)
            for tile_img, tile_rect in world.obstacle_list
        )

class GameActions(enum.IntEnum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    JUMP = 2
    SHOOT = 3
    GRENADE = 4
    STOP = 5
