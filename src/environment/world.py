from src.settings import *
from src.environment.entities import Decoration, Water, Exit, HealthBar, ItemBox
from src.environment.soldier import Soldier

TILE_SIZE_W = TILE_SIZE
TILE_SIZE_H = TILE_SIZE


class World():
    def __init__(self):
        self.obstacle_list = []
        self.world_data = []  # store the raw tile data for lookups
        self.exit_xy = {}
        self.tilescroll = 0

    def set_world_data(self, data):
        self.world_data = data

    # def process_data(self, data):
    #     global player
    #     player = None
    #     self.level_length = len(data[0])
    #
    #     # Store the map grid in the world instance
    #     self.set_world_data(data)
    #
    #     # Iterate through each value in level data file
    #     for y, row in enumerate(data):
    #         for x, tile in enumerate(row):
    #             if tile >= 0:
    #                 img = img_list[tile]
    #                 img_rect = img.get_rect()
    #                 img_rect.x = x * TILE_SIZE_W
    #                 img_rect.y = y * TILE_SIZE_H
    #                 tile_data = (img, img_rect)
    #                 if tile >= 0 and tile <= 8:
    #                     self.obstacle_list.append(tile_data)
    #                 elif tile >= 9 and tile <= 10:
    #                     water = Water(img, x * TILE_SIZE_W, y * TILE_SIZE_H)
    #                     water_group.add(water)
    #                 elif tile >= 11 and tile <= 14:
    #                     decoration = Decoration(img, x * TILE_SIZE_W, y * TILE_SIZE_H)
    #                     decoration_group.add(decoration)
    #                 elif tile == 15:
    #                     player = Soldier('player', x * TILE_SIZE_W, y * TILE_SIZE_H, 1.5, 5, 20, 5)
    #                     health_bar = HealthBar(10, 10, player.health, player.health)
    #                 elif tile == 16:
    #                     enemy = Soldier('enemy', x * TILE_SIZE_W, y * TILE_SIZE_H, 1.125, 2, 20, 0)
    #                     enemy_group.add(enemy)
    #                 elif tile == 17:
    #                     item_box = ItemBox('Ammo', x * TILE_SIZE_W, y * TILE_SIZE_H, player)
    #                     item_box_group.add(item_box)
    #                 elif tile == 18:
    #                     item_box = ItemBox('Grenade', x * TILE_SIZE_W, y * TILE_SIZE_H, player)
    #                     item_box_group.add(item_box)
    #                 elif tile == 19:
    #                     item_box = ItemBox('Health', x * TILE_SIZE_W, y * TILE_SIZE_H, player)
    #                     item_box_group.add(item_box)
    #                 elif tile == 20:
    #                     exit = Exit(img, x * TILE_SIZE_W, y * TILE_SIZE_H)
    #                     exit_group.add(exit)
    #                     self.exit_xy.update({
    #                         "x": x,
    #                         "y": y
    #                     })
    #
    #     return player, health_bar

    def process_data(self, data):
        global player
        player = None
        self.level_length = len(data[0])

        # Store the map grid in the world instance
        self.set_world_data(data)

        #iterate through each value in level data file
        for y, row in enumerate(data):
            for x, tile in enumerate(row):
                if tile >= 0:
                    img = img_list[tile]
                    img_rect = img.get_rect()
                    img_rect.x = x * TILE_SIZE
                    img_rect.y = y * TILE_SIZE
                    tile_data = (img, img_rect)
                    if tile >= 0 and tile <= 8:
                        self.obstacle_list.append(tile_data)
                    elif tile >= 9 and tile <= 10:
                        water = Water(img, x * TILE_SIZE, y * TILE_SIZE)
                        water_group.add(water)
                    elif tile >= 11 and tile <= 14:
                        decoration = Decoration(img, x * TILE_SIZE, y * TILE_SIZE)
                        decoration_group.add(decoration)
                    elif tile == 15:#create player
                        player = Soldier('player', x * TILE_SIZE, y * TILE_SIZE, 100, 1.5, 5, 50, 10)
                        health_bar = HealthBar(10, 10, player.health, player.health)
                    elif tile == 16:#create enemies
                        enemy = Soldier('enemy', x * TILE_SIZE, y * TILE_SIZE, 50,1.125, 2, 200, 0)
                        enemy_group.add(enemy)
                    elif tile == 17:#create ammo box
                        item_box = ItemBox('Ammo', x * TILE_SIZE, y * TILE_SIZE, player)
                        item_box_group.add(item_box)
                    elif tile == 18:#create grenade box
                        item_box = ItemBox('Grenade', x * TILE_SIZE, y * TILE_SIZE, player)
                        item_box_group.add(item_box)
                    elif tile == 19:#create health box
                        item_box = ItemBox('Health', x * TILE_SIZE, y * TILE_SIZE, player)
                        item_box_group.add(item_box)
                    elif tile == 20:#create exit
                        exit = Exit(img, x * TILE_SIZE, y * TILE_SIZE)
                        exit_group.add(exit)
        return player, health_bar

    def draw(self):
        for tile in self.obstacle_list:
            tile[1][0] += get_screen_scroll()
            screen.blit(tile[0], tile[1])

    def get_tile_coordinates(self, world_map, tile_range=(0, 8)):
        tile_coordinates = []
        for row in range(len(world_map)):
            for col in range(len(world_map[row])):
                if world_map[row][col] >= tile_range[0] and world_map[row][col] <= tile_range[1]:
                    # Calculate the top-left corner of the tile
                    x = col * TILE_SIZE_W  + self.tilescroll
                    y = row * TILE_SIZE_H
                    tile_coordinates.append((x, y))
        return tile_coordinates

    def get_empty_tile_coordinates(self, world_map, tile_range=(0, 8)):
        empty_tile_coordinates = []
        for row in range(len(world_map)):
            for col in range(len(world_map[row])):
                # If the tile is not in the specified range, it's considered empty
                if world_map[row][col] < tile_range[0] or world_map[row][col] > tile_range[1]:
                    # Calculate the top-left corner of the empty tile
                    x = col * TILE_SIZE_W + self.tilescroll
                    y = row * TILE_SIZE_H
                    empty_tile_coordinates.append((x, y))
        return empty_tile_coordinates

    def get_empty_spaces_between(self, world_map, tile_range=(0, 8)):
        # Get tile coordinates (within the range)
        tile_coordinates = self.get_tile_coordinates(world_map, tile_range)

        # Get empty tile coordinates (out of the range)
        empty_tile_coordinates = self.get_empty_tile_coordinates(world_map, tile_range)

        # Identify the empty spaces between these two
        empty_spaces = []
        # Go through each empty tile coordinate and check if it's not in the tile_coordinates
        for empty_tile in empty_tile_coordinates:
            if empty_tile not in tile_coordinates:
                empty_spaces.append(empty_tile)
        return empty_spaces

    def get_horizontal_empty_spaces(self, world_map, tile_range=(0, 8)):
        horizontal_empty_spaces = []

        for row in range(len(world_map)):
            for col in range(len(world_map[row]) - 1):  # Iterate until second-last column
                # Get the value of the current tile and the next tile in the row
                current_tile = world_map[row][col]
                next_tile = world_map[row][col + 1]

                # Check if there is a boundary between tiles: one is inside the range, the other is outside
                if (tile_range[0] <= current_tile <= tile_range[1] and not (
                        tile_range[0] <= next_tile <= tile_range[1])) or \
                        (not (tile_range[0] <= current_tile <= tile_range[1]) and tile_range[0] <= next_tile <=
                         tile_range[1]):
                    # Calculate the top-left corner of the empty space (horizontal gap between tiles)
                    x = (col + 1) * TILE_SIZE_W + self.tilescroll  # Position where the tile starts
                    y = row * TILE_SIZE_H  # Same y-coordinate for horizontal gaps
                    horizontal_empty_spaces.append((x, y))

        return horizontal_empty_spaces

    def get_player_tile_position(self, player_x, player_y):
        # Adjust the player's x position for the scrolling offset
        adjusted_x = player_x - self.tilescroll

        # Calculate the tile column and row
        tile_col = adjusted_x // TILE_SIZE_W
        tile_row = player_y // TILE_SIZE_H

        # Get the top-left corner of the tile
        tile_x = tile_col * TILE_SIZE_W + self.tilescroll
        tile_y = tile_row * TILE_SIZE_H

        return (tile_x, tile_y)

    def is_player_one_tile_before_empty_space(self, player_x, player_y, world_map, tile_range=(0, 8)):
        # Get all horizontal empty spaces
        empty_spaces = self.get_horizontal_empty_spaces(world_map, tile_range)

        # Get the player's current tile position
        player_tile_x, player_tile_y = self.get_player_tile_position(player_x, player_y)

        # Calculate the next tile to the right of the player
        next_tile_x = player_tile_x + TILE_SIZE_W

        # Check if this next tile is in the empty spaces
        for (empty_x, empty_y) in empty_spaces:
            # Check if the empty space is exactly one tile ahead
            if empty_x == next_tile_x and empty_y == player_tile_y:
                # Return True and the empty space position
                return True, (empty_x, empty_y)

        # If no empty space found, return False and None
        return False, []

    # def is_player_tile_ahead_below_or_back_below_empty_space(self, player_x, player_y, world_map,
    #                                                          tile_range=(0, 8)):
    #     # Get all horizontal empty spaces
    #     empty_spaces = self.get_horizontal_empty_spaces(world_map, tile_range)
    #
    #     # Get the player's current tile position
    #     player_tile_x, player_tile_y = self.get_player_tile_position(player_x, player_y)
    #
    #     # Calculate the next tile to the right, left, and below
    #     ahead_tile_x = player_tile_x + TILE_SIZE_W  # One tile ahead
    #     back_tile_x = player_tile_x - TILE_SIZE_W  # One tile back
    #     below_tile_y = player_tile_y + TILE_SIZE_H  # One tile below
    #
    #     empty_positions = []  # List to store the found empty positions
    #
    #     # Check if the empty space is one tile ahead and one tile below
    #     for (empty_x, empty_y) in empty_spaces:
    #         if empty_x == ahead_tile_x and empty_y == below_tile_y:
    #             empty_positions.append((empty_x, empty_y))  # Store the empty space position
    #             break  # Exit the loop once a valid position is found
    #
    #     # Check if the empty space is one tile back and one tile below
    #     for (empty_x, empty_y) in empty_spaces:
    #         if empty_x == back_tile_x and empty_y == below_tile_y:
    #             empty_positions.append((empty_x, empty_y))  # Store the empty space position
    #             break  # Exit the loop once a valid position is found
    #
    #     # If we found any empty positions, return True and the list of positions
    #     if empty_positions:
    #         return True, empty_positions
    #
    #     # If no empty space found, return False and None
    #     return False, []

    def is_player_tile_ahead_below_or_back_below_empty_space(self, player_x, player_y, world_map, tile_range=(0, 8)):
        # Get all horizontal empty spaces
        empty_spaces = self.get_horizontal_empty_spaces(world_map, tile_range)

        # Get the player's current tile position
        player_tile_x, player_tile_y = self.get_player_tile_position(player_x, player_y)

        # Calculate the next tile to the right, left, and below
        ahead_tile_x = player_tile_x + TILE_SIZE_W  # One tile ahead
        back_tile_x = player_tile_x - TILE_SIZE_W  # One tile back
        below_tile_y = player_tile_y + TILE_SIZE_H  # One tile below

        empty_positions = []  # List to store the found empty positions

        # Check if the empty space is one tile ahead and one tile below
        for (empty_x, empty_y) in empty_spaces:
            if empty_x == ahead_tile_x and empty_y == below_tile_y:
                empty_positions.append((empty_x, empty_y))  # Store the empty space position
                break  # Exit the loop once a valid position is found

        # Check if the empty space is one tile back and one tile below
        for (empty_x, empty_y) in empty_spaces:
            if empty_x == back_tile_x and empty_y == below_tile_y:
                empty_positions.append((empty_x, empty_y))  # Store the empty space position
                break  # Exit the loop once a valid position is found

        # If we found any empty positions, return True and the list of positions
        if empty_positions:
            return True, empty_positions

        # If no empty space found, return False and None
        return False, []

    def check_ahead_collision(self, player_rect, direction='right'):
        # Convert the player_rect tuple to a pygame.Rect object
        check_rect = player_rect.copy() # player_rect should be (x, y, width, height)

        # Create a check rectangle ahead of the player based on their direction
        if direction == 'right':
            check_rect.x += TILE_SIZE_W  # Move the check rectangle to the right

        # Check for collisions with obstacles
        for tile in self.obstacle_list:
            if tile[1].colliderect(check_rect):  # tile[1] is the rectangle of the obstacle
                return True  # There is a collision, unsafe to move

        return False  # No collision, safe to move

    # def get_gap_spaces(self, world_map, tile_range=(0, 8)):
    #     gaps = []
    #     rows = len(world_map)
    #     cols = len(world_map[0])
    #
    #     for row in range(rows - 1):  # skip last row to avoid index error
    #         for col in range(cols - 1):
    #             current_tile = world_map[row][col]
    #             right_tile = world_map[row][col + 1]
    #             below_tile = world_map[row + 1][col]
    #             diag_tile = world_map[row + 1][col + 1]
    #
    #             # Case 1: Horizontal gap — current tile is ground, right tile is empty
    #             if tile_range[0] <= current_tile <= tile_range[1] and not (
    #                     tile_range[0] <= right_tile <= tile_range[1]):
    #                 gap_start = col + 1
    #                 gap_end = gap_start
    #                 while gap_end < cols and not (tile_range[0] <= world_map[row][gap_end] <= tile_range[1]):
    #                     gap_end += 1
    #                 gap_width = (gap_end - gap_start) * TILE_SIZE_W
    #                 gap_x = gap_start * TILE_SIZE_W + self.tilescroll
    #                 gap_y = row * TILE_SIZE_H
    #                 gaps.append({
    #                     'type': 'horizontal',
    #                     'start_col': gap_start,
    #                     'end_col': gap_end - 1,
    #                     'width': gap_width,
    #                     'x': gap_x,
    #                     'y': gap_y
    #                 })
    #
    #             # Case 2: Diagonal gap — current tile is ground, lower-right is ground, but right & below are empty
    #             if (tile_range[0] <= current_tile <= tile_range[1] and
    #                     tile_range[0] <= diag_tile <= tile_range[1] and
    #                     not (tile_range[0] <= right_tile <= tile_range[1]) and
    #                     not (tile_range[0] <= below_tile <= tile_range[1])):
    #                 gap_x = (col + 1) * TILE_SIZE_W + self.tilescroll
    #                 gap_y = (row + 1) * TILE_SIZE_H
    #                 gaps.append({
    #                     'type': 'diagonal',
    #                     'start_col': col,
    #                     'end_col': col + 1,
    #                     'width': TILE_SIZE_W,
    #                     'x': gap_x,
    #                     'y': gap_y
    #                 })
    #     return gaps

    # def get_gap_spaces(self, world_map, tile_range=(0, 8)):
    #     gaps = []
    #     rows = len(world_map)
    #     cols = len(world_map[0])
    #
    #     for row in range(rows - 1):
    #         for col in range(cols - 1):
    #             current_tile = world_map[row][col]
    #             right_tile = world_map[row][col + 1]
    #             below_tile = world_map[row + 1][col]
    #             diag_tile = world_map[row + 1][col + 1]
    #
    #             # ---- Horizontal Gaps ----
    #             if tile_range[0] <= current_tile <= tile_range[1] and not (
    #                     tile_range[0] <= right_tile <= tile_range[1]):
    #                 gap_start = col + 1
    #                 gap_end = gap_start
    #                 while gap_end < cols and not (tile_range[0] <= world_map[row][gap_end] <= tile_range[1]):
    #                     gap_end += 1
    #                 gap_width = (gap_end - gap_start) * TILE_SIZE_W
    #                 gap_x = gap_start * TILE_SIZE_W + self.tilescroll
    #                 gap_y = row * TILE_SIZE_H
    #                 gaps.append({
    #                     'type': 'horizontal',
    #                     'start_col': gap_start,
    #                     'end_col': gap_end - 1,
    #                     'width': gap_width,
    #                     'x': gap_x,
    #                     'y': gap_y
    #                 })
    #
    #             # ---- Diagonal Gaps ----
    #             if (
    #                     tile_range[0] <= current_tile <= tile_range[1] and  # current is ground
    #                     tile_range[0] <= diag_tile <= tile_range[1] and  # bottom-right is ground
    #                     not (tile_range[0] <= right_tile <= tile_range[1]) and  # right is empty
    #                     not (tile_range[0] <= below_tile <= tile_range[1])  # below is empty
    #             ):
    #                 gap_x = (col + 1) * TILE_SIZE_W + self.tilescroll
    #                 gap_y = (row + 1) * TILE_SIZE_H
    #                 gaps.append({
    #                     'type': 'diagonal',
    #                     'start_col': col,
    #                     'end_col': col + 1,
    #                     'width': TILE_SIZE_W,
    #                     'x': gap_x,
    #                     'y': gap_y
    #                 })
    #
    #     return gaps

    def get_diagonal_tile_coordinates(self, tile_range=(0, 8)):
        coords_top_left_to_bottom_right = []
        coords_top_right_to_bottom_left = []
        coords_other_diagonals = []

        rows = len(self.world_data)
        cols = len(self.world_data[0]) if rows > 0 else 0

        for row in range(rows):
            for col in range(cols):
                tile = self.world_data[row][col]
                if tile_range[0] <= tile <= tile_range[1]:
                    # Top-left to Bottom-right diagonal (row == col)
                    if row == col:
                        x = col * TILE_SIZE_W + self.tilescroll
                        y = row * TILE_SIZE_H
                        coords_top_left_to_bottom_right.append((x, y))

                    # Top-right to Bottom-left diagonal (row + col = cols - 1)
                    if row + col == cols - 1:
                        x = col * TILE_SIZE_W + self.tilescroll
                        y = row * TILE_SIZE_H
                        coords_top_right_to_bottom_left.append((x, y))

                    # Other diagonals (you can define your own range or condition)
                    if abs(row - col) <= 1:  # Example for near-diagonal tiles
                        x = col * TILE_SIZE_W + self.tilescroll
                        y = row * TILE_SIZE_H
                        coords_other_diagonals.append((x, y))

        return {
            "top_left_to_bottom_right": coords_top_left_to_bottom_right,
            "top_right_to_bottom_left": coords_top_right_to_bottom_left,
            "other_diagonals": coords_other_diagonals
        }

    def raycast_check_should_jump(self, player_rect):
        ray_distance = TILE_SIZE_W  # how far ahead to check
        ray_height_offset = 5  # slight offset to keep ray above ground

        # Raycast point ahead of player (same vertical position as player's feet)
        ray_origin_x = player_rect.x + player_rect.width + ray_distance
        ray_origin_y = player_rect.y + player_rect.height - ray_height_offset

        # # Ray going straight down from the point ahead
        # ray_end_y = ray_origin_y + TILE_SIZE_H

        ground_detected = False
        obstacle_detected = False

        # Check if there's ground below the point ahead (simulate downward ray)
        ray_rect = pygame.Rect(ray_origin_x, ray_origin_y, 2, TILE_SIZE_H)
        for _, tile_rect in self.obstacle_list:
            if tile_rect.colliderect(ray_rect):
                ground_detected = True
                break

        # Check if there's an obstacle directly ahead at walking level
        forward_ray_rect = pygame.Rect(player_rect.x + player_rect.width, player_rect.y, TILE_SIZE_W,
                                       player_rect.height)
        for _, tile_rect in self.obstacle_list:
            if tile_rect.colliderect(forward_ray_rect):
                obstacle_detected = True
                break

        # Return True if a gap or an obstacle is detected
        if not ground_detected or obstacle_detected:
            return True

        return False

    def raycast_for_gap(self, player, tile_range=(0, 8), ray_length=100):
        # Get player's foot position
        player_feet_x = player.rect.centerx
        player_feet_y = player.rect.bottom

        # Cast ray forward (to the right)
        for dx in range(0, ray_length, TILE_SIZE):
            ray_x = player_feet_x + dx
            ray_y = player_feet_y + 5  # Slightly below feet

            # Determine tile coordinates
            tile_col = (ray_x - self.tilescroll) // TILE_SIZE_W
            tile_row = ray_y // TILE_SIZE_H

            if tile_row >= len(self.world_data) or tile_col >= len(self.world_data[0]):
                continue  # Out of bounds

            tile_value = self.world_data[tile_row][tile_col]

            # Check if the tile is empty (not ground)
            if tile_value < tile_range[0] or tile_value > tile_range[1]:
                # Draw the ray (for debug)
                pygame.draw.line(screen, (255, 0, 0), (player_feet_x, player_feet_y), (ray_x, ray_y), 2)
                return True  # Gap detected

            # Optional: Draw green if safe
            pygame.draw.line(screen, (0, 255, 0), (player_feet_x, player_feet_y), (ray_x, ray_y), 1)

        return False  # No gap ahead

    def raycast_for_gap2(self, player, tile_range=(0, 8), ray_length=400):
        # Get player's foot position
        player_feet_x = player.rect.centerx
        player_feet_y = player.rect.bottom

        # Cast ray forward (to the right) and upward (to the right upwards)
        for dx in range(0, ray_length, TILE_SIZE):
            # Forward ray (horizontal)
            ray_x = player_feet_x + dx
            ray_y = player_feet_y + 5  # Slightly below feet

            # Determine tile coordinates for forward ray
            tile_col = (ray_x - self.tilescroll) // TILE_SIZE_W
            tile_row = ray_y // TILE_SIZE_H

            if tile_row >= len(self.world_data) or tile_col >= len(self.world_data[0]):
                continue  # Out of bounds

            tile_value = self.world_data[tile_row][tile_col]

            # Check if the tile is empty (not ground)
            if tile_value < tile_range[0] or tile_value > tile_range[1]:
                # Draw the forward ray (for debug)
                pygame.draw.line(screen, (255, 0, 0), (player_feet_x, player_feet_y), (ray_x, ray_y), 2)
                return True  # Gap detected

            # Optional: Draw green if safe
            pygame.draw.line(screen, (0, 255, 0), (player_feet_x, player_feet_y), (ray_x, ray_y), 1)

            # Right upward ray (diagonal upwards to the right)
            ray_x_up = player_feet_x + dx
            ray_y_up = player_feet_y - dx  # Move upward as we move right

            # Determine tile coordinates for upward ray
            tile_col_up = (ray_x_up - self.tilescroll) // TILE_SIZE_W
            tile_row_up = ray_y_up // TILE_SIZE_H

            if tile_row_up >= len(self.world_data) or tile_col_up >= len(self.world_data[0]):
                continue  # Out of bounds

            tile_value_up = self.world_data[tile_row_up][tile_col_up]

            # Check if the tile above is not ground (gap)
            if tile_value_up < tile_range[0] or tile_value_up > tile_range[1]:
                # Draw the right upward ray (for debug)
                pygame.draw.line(screen, (255, 0, 0), (player_feet_x, player_feet_y), (ray_x_up, ray_y_up), 2)
                return True  # Gap detected

            # Optional: Draw green if safe
            pygame.draw.line(screen, (0, 255, 0), (player_feet_x, player_feet_y), (ray_x_up, ray_y_up), 1)

        return False  # No gap ahead



















