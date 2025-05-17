import math
import os
import random
from collections import deque

import numpy as np
from pygame import Vector2

# from src.environment.world import World
from src.settings import *
from src.environment.equipments import Bullet


class Soldier(pygame.sprite.Sprite):
    def __init__(self, char_type, x, y, health, scale, speed, ammo, grenades):
        pygame.sprite.Sprite.__init__(self)
        self.alive = True
        self.coin = 0
        self.char_type = char_type
        self.speed = speed
        self.ammo = ammo
        self.max_ammo = ammo
        self.start_ammo = ammo
        self.shoot_cooldown = 0
        self.grenades = grenades
        self.max_grenades = grenades
        self.health = health
        self.max_health = self.health
        self.direction = 1
        self.vel_y = 0
        self.vel_x = 0
        self.jump = False
        self.in_air = True
        self.flip = False
        self.animation_list = []
        self.frame_index = 0
        self.action = 0
        self.update_time = pygame.time.get_ticks()
        # ai specific variables
        self.move_counter = 0
        self.vision = pygame.Rect(0, 0, 150, 20)
        self.idling = False
        self.idling_counter = 0
        self.prev_x = x  # Track previous position
        self.moved_forward = False
        self.moved_backward = False
        self.bullets_hit_this_frame = 0  # Track successful hits
        # Add health tracking
        self.prev_health = health  # Initialize with max health
        self.velocity_history = deque(maxlen=10)  # Track last 10 frames

        # load all images for the players
        animation_types = ['Idle', 'Run', 'Jump', 'Death']
        for animation in animation_types:
            # reset temporary list of images
            temp_list = []
            # count number of files in the folder
            num_of_frames = len(os.listdir(f'src/assets/images/{self.char_type}/{animation}')) - 1
            for i in range(num_of_frames):
                img = pygame.image.load(f'src/assets/images/{self.char_type}/{animation}/{i}.png').convert_alpha()
                img = pygame.transform.scale(img, (int(img.get_width() * scale), int(img.get_height() * scale)))
                img = clip_image(img, 0.26, 0.36)
                temp_list.append(img)
            self.animation_list.append(temp_list)

        self.image = self.animation_list[self.action][self.frame_index]
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.width = self.image.get_width()
        self.height = self.image.get_height()
        self.last_x = self.rect.x
        self.moved_one_tile = None
        self.reach_exit = False

    def update(self):
        self.update_animation()
        self.check_alive()
        # update cooldown
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        # Store health state before updates
        if self.char_type == 'player':
            self.prev_health = self.health

    def move(self, moving_left, moving_right, world):
        # reset movement variables
        screen_scroll = 0
        dx = 0
        dy = 0

        # assign movement variables if moving left or right
        # if moving_left:
        #     dx = -self.speed - 1.8
        #     self.flip = True
        #     self.direction = -1
        # if moving_right:
        #     dx = self.speed + 1.8
        #     self.flip = False
        #     self.direction = 1

        if moving_left:
            dx = -self.speed
            self.flip = True
            self.direction = -1
        if moving_right:
            dx = self.speed
            self.flip = False
            self.direction = 1

        # jump
        if self.jump == True and self.in_air == False:
            self.vel_y = -12.65 #14 .375
            self.vel_x = 3.5 * self.direction
            self.jump = False
            self.in_air = True

        # apply gravity
        self.vel_y += GRAVITY
        if self.vel_y > 10:
            self.vel_y

        dy += self.vel_y

        # check if player is on the ground
        if self.vel_y == 0:
            self.vel_y = 0
            self.in_air = False


        # check for collision
        for tile in world.obstacle_list:
            # check collision in the x direction
            if tile[1].colliderect(self.rect.x + dx, self.rect.y, self.width, self.height):
                dx = 0
                # if the ai has hit a wall then make it turn around
                if self.char_type == 'enemy':
                    self.direction *= -1
                    self.move_counter = 0
            # check for collision in the y direction
            if tile[1].colliderect(self.rect.x, self.rect.y + dy, self.width, self.height):
                # check if below the ground, i.e. jumping
                if self.vel_y < 0:
                    self.vel_y = 0
                    dy = tile[1].bottom - self.rect.top
                # check if above the ground, i.e. falling
                elif self.vel_y >= 0:
                    self.vel_y = 0
                    self.in_air = False
                    dy = tile[1].top - self.rect.bottom

        # check for collision with water
        if pygame.sprite.spritecollide(self, water_group, False):
            self.health = 0

        # check for collision with exit
        level_complete = False
        if pygame.sprite.spritecollide(self, exit_group, False):
            level_complete = True

        # check if fallen off the map
        if self.rect.bottom > SCREEN_HEIGHT:
            self.health = 0

        # check if going off the edges of the screen
        if self.char_type == 'player':
            if self.rect.left + dx < 0 or self.rect.right + dx > SCREEN_WIDTH:
                dx = 0

        # update rectangle position
        self.rect.x += dx
        self.rect.y += dy

        # update scroll based on player position
        if self.char_type == 'player':
            if (self.rect.right > SCREEN_WIDTH - SCROLL_THRESH and bg_scroll < (
                    world.level_length * TILE_SIZE) - SCREEN_WIDTH) \
                    or (self.rect.left < SCROLL_THRESH and bg_scroll > abs(dx)):
                self.rect.x -= dx
                screen_scroll = -dx

        # After updating player position
        self.vel_x = self.rect.x - self.last_x
        self.last_x = self.rect.x

        # Update movement tracking
        if self.char_type == 'player':
            self.moved_forward = (self.rect.x + 25> self.prev_x and self.direction == 1)
            self.moved_backward = (self.rect.x - 3< self.prev_x and self.direction == -1)

        self.prev_x = self.rect.x

        # Final ground check to correct in_air state
        self.in_air = True
        for tile in world.obstacle_list:
            if tile[1].colliderect(self.rect.x, self.rect.bottom + 1, self.width, 1):
                self.in_air = False
                break

        return screen_scroll, level_complete

    def shoot(self):
        if self.shoot_cooldown == 0 and self.ammo > 0:
            self.shoot_cooldown = 20
            bullet = Bullet(self.rect.centerx + (0.75 * self.rect.size[0] * self.direction), self.rect.centery,
                            self.direction)
            bullet_group.add(bullet)
            # reduce ammo
            self.ammo -= 1
            shot_fx.play()

    def bullet_hit_enemy(self):
        self.bullets_hit_this_frame = 0  # Reset counter
        # Check for new collisions
        for enemy in enemy_group:
            if pygame.sprite.spritecollide(enemy, bullet_group, True):
                self.bullets_hit_this_frame += 1
        return self.bullets_hit_this_frame > 0



    def get_nearest_enemy(self, enemy_group, radius=100):
        nearest_enemy = None
        min_distance = float('inf')

        for enemy in enemy_group:
            distance = np.sqrt((self.rect.centerx - enemy.rect.centerx) ** 2 +
                               (self.rect.centery - enemy.rect.centery) ** 2)
            if distance < radius and distance < min_distance:
                min_distance = distance
                nearest_enemy = enemy

        return nearest_enemy

    def fell_or_hit_water(self):
        in_water = pygame.sprite.spritecollide(self, water_group, False)
        fell_off = self.rect.bottom > SCREEN_HEIGHT
        if in_water or fell_off:
            self.alive = False
        return in_water or fell_off

    def distance_to_exit(self):
        player_center = self.rect.center
        min_distance = float('inf')

        for exit in exit_group:
            exit_center = exit.rect.center
            distance = abs(exit_center[0] - player_center[0]) + abs(exit_center[1] - player_center[1])
            min_distance = min(min_distance, distance)

        return min_distance

    def exit_coordinates(self):
        for exit in exit_group:
            return exit.rect.center
        return (0, 0)

    def reached_exit(self):
        return any(self.rect.colliderect(exit.rect) for exit in exit_group)

    def walked_forward(self):
        return self.moved_forward  # Use the tracked movement state

    def has_moved_one_tile_directional(self,  tile_size=5):
        delta = self.rect.x - self.prev_x
        print("---------player speed: ", self.speed , delta, TILE_SIZE, tile_size)
        if delta >= tile_size:
            return 1  # right
        elif delta <= -tile_size:
            return -1  # left
        else:
            return 0  # stand

    def can_move_one_tile(player, world, direction="right", tile_size=32):
        if direction == "left":
            target_x = player.rect.left - tile_size
        elif direction == "right":
            target_x = player.rect.right + tile_size - player.rect.width
        else:
            raise ValueError("Direction must be 'left' or 'right'.")

        # Define a small box under the intended position to check for ground
        check_rect = pygame.Rect(target_x, player.rect.bottom + 5, player.rect.width, 5)

        for tile in world.obstacle_list:
            if tile[1].colliderect(check_rect):
                return True  # Safe to move (ground exists)
        return False  # No ground, unsafe

    def player_near_edge(self, world):
        # Check if the player is near the edge
        player_left_edge = self.rect.left - 10
        player_right_edge = self.rect.right + 10
        is_near_edge = True

        # Check for tiles beneath the player's left and right edges
        for tile in world.obstacle_list:
            # Check for ground support on the left and right edges
            if tile[1].colliderect(player_left_edge, self.rect.bottom + 5, 5, 5) or \
                    tile[1].colliderect(player_right_edge, self.rect.bottom + 5, 5, 5):
                is_near_edge = False
                break
        return  is_near_edge

    def player_near_edge2(self, world):
        # Small padding to check left and right edges
        left_check_x = self.rect.left - 10
        right_check_x = self.rect.right + 10
        bottom_y = self.rect.bottom + 5  # Small buffer below the player

        # Assume the player is near an edge
        is_near_edge = True

        # Check only nearby tiles for efficiency
        for tile in world.obstacle_list:
            tile_rect = tile[1]

            # Check if there is ground support at the left or right side
            if tile_rect.colliderect(left_check_x, bottom_y, 5, 5) or \
                    tile_rect.colliderect(right_check_x, bottom_y, 5, 5):
                is_near_edge = False
                break

        return is_near_edge

    def is_near_edge(self, world):
        if world is None:
            return False

        # Check the tile directly under the player
        player_bottom = self.rect.bottom
        left_edge = self.rect.left - 5
        right_edge = self.rect.right + 5

        on_left_edge = not any(
            tile[1].colliderect(left_edge, player_bottom, 1, 1) for tile in world.obstacle_list
        )
        on_right_edge = not any(
            tile[1].colliderect(right_edge, player_bottom, 1, 1) for tile in world.obstacle_list
        )

        return on_left_edge or on_right_edge

    def walked_backward(self):
        return self.moved_backward

    def ai(self, player=None, world=None):
        # Add movement tracking for NPCs
        self.prev_x = self.rect.x  # Store position before moving
        if self.alive and player.alive:
            if self.idling == False and random.randint(1, 500) == 1:
                self.update_action(0)  # 0: idle
                self.idling = True
                self.idling_counter = 100
            # check if the ai in near the player
            if self.vision.colliderect(player.rect):
                # stop running and face the player
                self.update_action(0)  # 0: idle
                # shoot
                self.shoot()
            else:
                if self.idling == False:
                    if self.direction == 1:
                        ai_moving_right = True
                    else:
                        ai_moving_right = False
                    ai_moving_left = not ai_moving_right
                    self.move(ai_moving_left, ai_moving_right, world)
                    self.update_action(1)  # 1: run
                    self.move_counter += 1
                    # update ai vision as the enemy moves
                    self.vision.center = (self.rect.centerx + 95 * self.direction, self.rect.centery)

                    if self.move_counter > TILE_SIZE:
                        self.direction *= -1
                        self.move_counter *= -1
                else:
                    self.idling_counter -= 1
                    if self.idling_counter <= 0:
                        self.idling = False

        # scroll
        self.rect.x += get_screen_scroll()

        # # After movement:
        # if self.rect.x != self.prev_x:
        #     self.moved_forward = (self.rect.x > self.prev_x and self.direction == 1) or \
        #                          (self.rect.x < self.prev_x and self.direction == -1)

    def update_animation(self):
        # update animation
        ANIMATION_COOLDOWN = 100
        # update image depending on current frame
        self.image = self.animation_list[self.action][self.frame_index]
        # check if enough time has passed since the last update
        if pygame.time.get_ticks() - self.update_time > ANIMATION_COOLDOWN:
            self.update_time = pygame.time.get_ticks()
            self.frame_index += 1
        # if the animation has run out the reset back to the start
        if self.frame_index >= len(self.animation_list[self.action]):
            if self.action == 3:  # 3: death
                self.frame_index = len(self.animation_list[self.action]) - 1
            else:
                self.frame_index = 0

    def update_action(self, new_action):
        # check if the new action is different to the previous one
        if new_action != self.action:
            self.action = new_action
            # update the animation settings
            self.frame_index = 0
            self.update_time = pygame.time.get_ticks()

    def check_alive(self):
        if self.health <= 0:
            self.health = 0
            self.speed = 0
            self.alive = False
            self.update_action(3)

    def draw(self):
        screen.blit(pygame.transform.flip(self.image, self.flip, False), self.rect)

        if DEBUG_SHOW_COLLISION_BOX:
            # Debug: Draw red rectangle around the collision box
            pygame.draw.rect(screen, (255, 0, 0), self.rect, 2)
