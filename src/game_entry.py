import math
import os
import random
import time
from collections import deque

import cv2
import numpy as np
import pygame
import tensorflow
import torch
from matplotlib import pyplot as plt
from pygame import mixer, Vector2
from pygame._sdl2 import Image
from rich import region
from tensorflow.python.keras.backend_config import epsilon

from src.graph.plot import TrainingVisualizer

mixer.init()
pygame.init()

from src.settings import *
from src.environment.entities import ScreenFade
from src.environment.world import World
from src.environment.equipments import Grenade
from src.ai_agent.agent_state_and_action import ExtractGameState, GameActions, ImageExtractorThread
from src.ai_agent.agent import DQNAgent
from src.ai_agent.reward import RewardAI
from src.ai_agent.save_model_data import SaveFutureLearning
from src.utils.logger import TrainingLogger
import threading
import pygame
import time

run = True
save_data = True
done = False
game_start_time = time.time()

intro_fade = ScreenFade(1, BLACK, 4)
death_fade = ScreenFade(2, PINK, 4)

# Global flags
world = World()
player, health_bar = world.process_data(get_world_data(0))


def timer():
    global game_start_time, done
    current_time = time.time()
    if current_time - game_start_time > 30:
        done = True


def show_intro_fade():
    global intro_fade, start_intro
    # show intro
    if start_intro == True:
        if intro_fade.fade():
            start_intro = False
            intro_fade.fade_counter = 0


def create_start_game_menu():
    global run, start_game, start_intro
    # draw menu
    screen.fill(BG)
    # add buttons
    if start_button.draw(screen):
        start_game = True
        start_intro = True
    if exit_button.draw(screen):
        run = False

# Add this with your other global variables
level_start_time = pygame.time.get_ticks()  # Initialize timer when level starts
# Add these global variables
start_x = 0
exit_x = SCREEN_WIDTH  # Default to screen width if no exit

def update_game_menu():
    global world, player, health_bar, grenade, epsilon, level, start_x, exit_x, episode, total_reward, life_line, chosen_action, action_type
    # update background
    draw_bg()
    # draw world map
    world.draw()
    # show player health
    health_bar.draw(player.health)
    # show ammo with image and count
    draw_text('AMMO: ', font, WHITE, 10, 35)
    screen.blit(bullet_img, (95, 43))
    draw_text(f' {player.ammo}', font, WHITE, 110, 35)
    # show grenades with image and count
    draw_text('GRENADES: ', font, WHITE, 10, 60)
    screen.blit(grenade_img, (135, 65))
    draw_text(f' {player.grenades}', font, WHITE, 155, 60)
    # show grenades with image and count
    draw_text('COINS: ', font, WHITE, 10, 85)
    coin_box_img2 = pygame.transform.scale(coin_box_img, (20, 20))
    screen.blit(coin_box_img2, (90, 85))
    draw_text(f' {player.coin}', font, WHITE, 125, 83)


    # Calculate elapsed time
    current_time = pygame.time.get_ticks()
    elapsed_time = (current_time - level_start_time) // 1000  # Convert to seconds
    minutes = elapsed_time // 60
    seconds = elapsed_time % 60
    time_str = f"{minutes:02}:{seconds:02}"  # Format as MM:SS

    # Show game timer in top-right corner
    # Show game timer and level in top-right corner  / {MAX_LEVELS}
    draw_text(f'LEVEL: {level}', font, WHITE, SCREEN_WIDTH - 150, 10)
    draw_text(f'TIME: {time_str}', font, WHITE, SCREEN_WIDTH - 150, 40)
    # draw_text(f'TIME: {time_str}', font, WHITE, SCREEN_WIDTH - 150, 10)

    # Get level boundaries once
    if 'start_x' not in globals():
        start_x = player.rect.x
    if exit_group:
        exit_x = exit_group.sprites()[0].rect.x if exit_group else SCREEN_WIDTH

    # Calculate progress percentage
    max_distance = exit_x - start_x
    current_progress = player.rect.x - start_x
    progress_pct = (current_progress / max_distance) * 100 if max_distance > 0 else 0
    progress_pct = max(0, min(100, progress_pct))  # Clamp between 0-100

    # --- Compact Progress Display ---
    # Position under timer (assuming timer is at SCREEN_WIDTH - 150, 10)
    display_x = SCREEN_WIDTH - 160  # 10px left of timer
    display_y = 70  # 30px below time

    # Progress Bar
    progress_bar_rect = pygame.Rect(display_x, display_y, 150, 8)
    pygame.draw.rect(screen, WHITE, progress_bar_rect, 1)  # Border
    fill_width = int(150 * (progress_pct / 100))
    pygame.draw.rect(screen, GREEN, (display_x, display_y, fill_width, 8))

    # Mini Map
    mini_map_rect = pygame.Rect(display_x, display_y + 15, 150, 40)  # 15px below progress bar
    pygame.draw.rect(screen, WHITE, mini_map_rect, 1)  # Border

    # Start/End markers (smaller)
    pygame.draw.line(screen, RED,
                     (mini_map_rect.left + 3, mini_map_rect.centery),
                     (mini_map_rect.left + 3, mini_map_rect.centery + 6), 2)
    pygame.draw.line(screen, BLUE,
                     (mini_map_rect.right - 3, mini_map_rect.centery),
                     (mini_map_rect.right - 3, mini_map_rect.centery + 6), 2)

    # Player position indicator
    if max_distance > 0:
        player_pos_x = mini_map_rect.left + 3 + (current_progress / max_distance) * (mini_map_rect.width - 6)
        pygame.draw.circle(screen, YELLOW, (int(player_pos_x), mini_map_rect.centery), 2)
    # pygame.time.wait(300)
    if not DEBUG:
        draw_text(f'Episode: {episode}', font, WHITE, SCREEN_WIDTH - 480, 20)
        draw_text(f'Total Reward:{total_reward:.2f}', font, WHITE, SCREEN_WIDTH - 480, 40)
        draw_text(f'Epsilon:{epsilon:.4f}', font, WHITE, SCREEN_WIDTH - 480, 60)
        draw_text(f'Action Type:{action_type}', font, WHITE, SCREEN_WIDTH - 480, 80)
        draw_text(f'Action:{GameActions(chosen_action).name}', font, WHITE, SCREEN_WIDTH - 480,100)


def update_game():
    global run, start_game, world, player, health_bar, intro_fade, death_fade, shoot, grenade, level, moving_left, moving_right, bg_scroll, grenade_thrown
    # update and draw player
    player.update()
    player.draw()

    # update and draw enemy
    for enemy in enemy_group:
        enemy.ai(player, world)
        enemy.update()
        enemy.draw()

    # update and draw groups
    bullet_group.update(player, world)
    grenade_group.update(player, world)
    explosion_group.update()
    item_box_group.update(player)
    decoration_group.update()
    water_group.update()
    exit_group.update()

    bullet_group.draw(screen)
    grenade_group.draw(screen)
    explosion_group.draw(screen)
    item_box_group.draw(screen)
    decoration_group.draw(screen)
    water_group.draw(screen)
    exit_group.draw(screen)

    # died = not player.alive
    # reached_exit = player.reached_exit() if hasattr(player, "reached_exit") else False
    # one_tile = player.has_moved_one_tile_directional(TILE_SIZE) == 1


def update_player_action():
    global run, start_game, start_intro, start_x, exit_x, level_start_time, world, player, health_bar, intro_fade, death_fade, shoot, grenade, level, moving_left, moving_right, bg_scroll, grenade_thrown

    # update player actions
    if player.alive:
        # shoot bullets
        if shoot:
            player.shoot()
        # throw grenades
        elif grenade and grenade_thrown == False and player.grenades > 0:
            grenade = Grenade(player.rect.centerx + (0.5 * player.rect.size[0] * player.direction),
                              player.rect.top, player.direction)
            grenade_group.add(grenade)
            # reduce grenades
            player.grenades -= 1
            grenade_thrown = True
        if player.in_air:
            player.update_action(2)  # 2: jump
        elif moving_left or moving_right:
            player.update_action(1)  # 1: run
        else:
            player.update_action(0)  # 0: idle
        screen_scroll, level_complete = player.move(moving_left, moving_right, world)
        set_screen_scroll(screen_scroll)
        bg_scroll -= get_screen_scroll()
        # check if player has completed the level
        if level_complete:
            start_intro = True
            # level += 1
            bg_scroll = 0
            # world_data = reset_level()
            # if level <= MAX_LEVELS:
            #     world = World()
            #     player, health_bar = world.process_data(world_data)
            # Reset the world for the new level
            world_data = get_world_data(level)
            world = World()
            player, health_bar = world.process_data(world_data)

            # Reset player position and level boundaries
            start_x = player.rect.x
            exit_x = exit_group.sprites()[0].rect.x if exit_group else SCREEN_WIDTH

            # Reset the level start time
            level_start_time = pygame.time.get_ticks()
    else:
        screen_scroll = 0
        set_screen_scroll(screen_scroll)
        bg_scroll -= get_screen_scroll()
        reset_game()


def update():
    global start_game

    # From here game runs
    if start_game == False:
        # Create Start Game Menu
        create_start_game_menu()
    else:
        # Create Top Game Menu
        update_game_menu()

        # Update Game
        update_game()

        # show intro
        # show_intro_fade()

        # update player actions
        update_player_action()


def reset_game(from_agent_click=False):
    global run, done, save_data, start_game, level_start_time, world, start_x, exit_x, player, health_bar, intro_fade, death_fade, shoot, grenade, level, moving_left, moving_right, bg_scroll, grenade_thrown, start_intro
    if death_fade.fade():
        if restart_button.draw(screen) or from_agent_click or True:
            save_data = True
            death_fade.fade_counter = 0
            start_intro = True
            bg_scroll = 0
            world_data = reset_level()
            world = World()
            player, health_bar = world.process_data(world_data)
            # start_x = player.rect.x
            # if exit_group:
            #     exit_x = exit_group.sprites()[0].rect.x
            # Add this with your other global variables
            level_start_time = pygame.time.get_ticks()  # Initialize timer when level starts
            # Add these global variables
            start_x = 0
            exit_x = SCREEN_WIDTH  # Default to screen width if no exit
    done = False


def perform_action(action):
    global moving_left, moving_right, shoot, grenade, grenade_thrown, player

    if action == GameActions.STOP:
        moving_right = False
        moving_left = False
        player.jump = True

    # Handle movement actions
    if action == GameActions.MOVE_LEFT:
        moving_left = True
        moving_right = False
    elif action == GameActions.MOVE_RIGHT:
        moving_right = True
        moving_left = False

    # Handle other actions
    shoot = (action == GameActions.SHOOT)

    if action == GameActions.JUMP:
        if player.alive and not player.in_air and player.moved_forward:
            player.jump = True
            jump_fx.play()

    if action == GameActions.GRENADE:
        grenade = True  # Let update_player_action handle the rest


# def perform_action(action):
#     # Simulate key press actions
#     if action == GameActions.MOVE_LEFT:
#         pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a))
#     elif action == GameActions.MOVE_RIGHT:
#         pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d))
#     elif action == GameActions.JUMP:
#         pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w))
#     elif action == GameActions.SHOOT:
#         pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
#     elif action == GameActions.GRENADE:
#         pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q))
#     else:
#         # Handle unexpected actions
#         print(f"Unknown action: {action}")
# keys = {
#     GameActions.MOVE_LEFT: pygame.K_a,
#     GameActions.MOVE_RIGHT: pygame.K_d,
#     GameActions.JUMP: pygame.K_w,
#     GameActions.SHOOT: pygame.K_SPACE,
#     GameActions.GRENADE: pygame.K_q
# }
# if action in keys:
#     pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=keys[action]))

TILE_SIZE_W = TILE_SIZE
TILE_SIZE_H = TILE_SIZE

player_spon_point = {
    "x": 0,
    "y": 0
}
life_line = 3
next_life = True

# def jump_logic():
#     global world, player, level
#     success = False
#     # Draw the empty tiles, adjusted for camera offset
#     empty_tiles = world.get_horizontal_empty_spaces(get_world_data(level))
#     cx, cy = world.get_player_tile_position(player.rect.centerx, player.rect.centery)
#     _, empty_positionx = world.is_player_tile_ahead_below_or_back_below_empty_space(cx, cy, get_world_data(1))
#     # Check if any empty tile is in the empty_positionx list
#     if len(empty_positionx):
#         for (tile_x, tile_y) in empty_tiles:
#             for (pos_x, pos_y) in empty_positionx:
#                 if tile_x == pos_x and tile_y == pos_y:
#                     success = True
#                     break  # Exit the inner loop once a collision is found
#     # empty_tile_colors = {tile: (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for tile
#     #                      in
#     #                      empty_tiles}
#     # for (x, y) in empty_tiles:
#     #     # pygame.draw.rect(screen, empty_tile_colors[(x, y)], (x, y, TILE_SIZE_W, TILE_SIZE_H))
#     #     empty_tile = pygame.Surface((TILE_SIZE_W, TILE_SIZE_H))
#     #     empty_tile.fill(empty_tile_colors[(x, y)])
#     #     empty_tile.set_alpha(128)
#     #     screen.blit(empty_tile, (x, y))
#     # # pygame.draw.rect(screen, (255, 255, 255), (cx, cy, TILE_SIZE_W, TILE_SIZE_H))
#     # player_c = pygame.Surface((TILE_SIZE_W, TILE_SIZE_H))
#     # player_c.fill((255, 255, 255))
#     # player_c.set_alpha(128)
#     # screen.blit(player_c, (cx, cy))
#
#     # if empty_positionx != None:
#     #     for (x, y) in empty_positionx:
#     #         # Only draw tiles within the visible screen area
#     #         # pygame.draw.rect(screen, (0, 0, 0), (x, y, TILE_SIZE_W, TILE_SIZE_H))
#     #         pygame.draw.rect(screen, (0, 0, 0), (x, y, TILE_SIZE_W, TILE_SIZE_H))
#     #         empty_positionx_color = pygame.Surface((TILE_SIZE_W, TILE_SIZE_H))
#     #         empty_positionx_color.fill((255, 255, 255))
#     #         empty_positionx_color.set_alpha(128)
#     #         screen.blit(empty_positionx_color, (x, y))
#     return success, (cx, cy), empty_tiles, (empty_positionx)

def manual_play():
    global run, start_game, world, player, epsilon, health_bar, intro_fade, death_fade, shoot, grenade, level, moving_left, moving_right, bg_scroll, grenade_thrown, start_intro, save_data, done
    # extract_state = ExtractGameState()
    iteration = 0

    while run:
        clock.tick(FPS)
        update()
        world.tilescroll += get_screen_scroll()
        # jump_logic()



        # Update the display
        pygame.display.flip()

        iteration = iteration + 1
        # Event Handling
        for event in pygame.event.get():
            # quit game
            if event.type == pygame.QUIT:
                run = False
            # keyboard presses
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    moving_left = True
                if event.key == pygame.K_d:
                    moving_right = True
                if event.key == pygame.K_SPACE:
                    shoot = True
                if event.key == pygame.K_q:
                    grenade = True
                if event.key == pygame.K_w and player.alive:
                    player.jump = True
                    jump_fx.play()
                if event.key == pygame.K_ESCAPE:
                    run = False
            # keyboard button released
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_a:
                    moving_left = False
                if event.key == pygame.K_d:
                    moving_right = False
                if event.key == pygame.K_SPACE:
                    shoot = False
                if event.key == pygame.K_q:
                    grenade = False
                    grenade_thrown = False

        pygame.display.update()
    pygame.quit()


def extract_additional_features():
    global enemy_group, world, player
    # Near Edge Detection (Assuming player x, y is top-left corner)
    near_edge = 1 if player.player_near_edge2(world) else 0
    move_ahead = player.walked_forward()
    alive = player.alive

    # Low Health
    low_health = 1 if player.health / player.max_health < 0.3 else 0

    # Distance to Nearest Enemy
    min_enemy_distance = float('inf')
    for enemy in enemy_group:
        enemy_center_x, enemy_center_y = enemy.rect.centerx, enemy.rect.centery
        enemy_distance = math.sqrt(
            (enemy_center_x - player.rect.centerx) ** 2 + (enemy_center_y - player.rect.centery) ** 2)
        min_enemy_distance = min(min_enemy_distance, enemy_distance)
    return np.array([move_ahead, alive, near_edge, low_health, min_enemy_distance])

# def extract_additional_features():
#     global player, enemy_group, world
#
#     # Exit Position
#     exit_coordinates = player.exit_coordinates()
#     exit_position_x, exit_position_y = exit_coordinates
#
#     # Current Position
#     current_position_x, current_position_y = player.rect.x, player.rect.y
#
#     # Distance to Exit
#     distance_to_exit = math.sqrt((exit_position_x - current_position_x) ** 2 + (exit_position_y - current_position_y) ** 2)
#
#     # Near Edge Detection
#     near_edge = 1 if player.player_near_edge(world) else 0
#
#     # Movement Status
#     move_ahead = player.walked_forward()
#     alive = 1 if player.alive else 0
#
#     # Low Health
#     low_health = 1 if player.health / player.max_health < 0.3 else 0
#
#     # Falling Speed
#     falling_speed = 1 if player.fell_or_hit_water() else 0
#
#     print(f"move_ahead: {move_ahead}, alive: {alive}, near_edge: {near_edge}, "
#           f"low_health: {low_health}, falling_speed: {falling_speed}, distance_to_exit: {distance_to_exit:.2f}")
#
#     # Combine all the features into a vector
#     return np.array([move_ahead, alive, near_edge, low_health, falling_speed, distance_to_exit])
#
#
# def extract_additional_features(player_jump):
#     global player, enemy_group, world
#     # Near Edge Detection (Assuming player x, y is top-left corner)
#     near_edge = 1 if player.player_near_edge2(world) else 0
#     move_ahead = player.walked_forward()
#     alive = player.alive
#
#     # Low Health
#     low_health = 1 if player.health / player.max_health < 0.3 else 0
#
#     # Enemy Nearby (within a certain radius)
#     # enemy_nearby = 1 if player.get_nearest_enemy(enemy_group) else 0
#
#     # # Falling Speed
#     falling_speed = 1 if player.fell_or_hit_water() else 0
#
#     # Distance to Nearest Pit (Water)
#     min_pit_distance = float('inf')
#     for water in water_group:
#         water_center_x, water_center_y = water.rect.centerx, water.rect.centery
#         pit_distance = math.sqrt(
#             (water_center_x - player.rect.x) ** 2 + (water_center_y - player.rect.y) ** 2)
#         min_pit_distance = min(min_pit_distance, pit_distance)
#
#     if player_jump is not None:
#
#         # Convert the boolean to a binary value
#         is_active = 1 if player_jump[0] else 0
#
#         # Convert the single coordinate to a 2-element array
#         position = np.array(player_jump[1])
#         # empty_tiles = np.array(player_jump[2]) #[:10]
#         single_tile = np.array(player_jump[3]) #[0]
#
#         # print("single_tile: ", single_tile)
#
#         is_active_array = np.array([is_active], dtype=np.float32)
#         position_array = np.array(position, dtype=np.float32)
#         # Flatten empty tiles if not empty
#         # if len(empty_tiles) > 0:
#         #     empty_tiles_array = np.array(empty_tiles, dtype=np.float32).flatten()
#         # else:
#         #     empty_tiles_array = np.array([], dtype=np.float32)
#
#         # Flatten single tiles if not empty
#         # if len(single_tile) > 0:
#         #     single_tile_array = np.array(single_tile, dtype=np.float32).flatten()
#         # else:
#         #     single_tile_array = np.array([], dtype=np.float32)
#
#         # nn_input = np.concatenate([[is_active], position, empty_tiles.flatten(), single_tile])
#         combined_features = np.concatenate([is_active_array, position_array])
#         nn_input_tensor = torch.tensor(combined_features, dtype=torch.float32)
#
#         # Print for Debugging
#         # print(f"move_ahead: {move_ahead}, alive: {alive}, near_edge: {near_edge}, "
#         #       f"low_health: {low_health}, falling_speed: {falling_speed}, "
#         #       f"min_pit_distance: {min_pit_distance:.2f}")
#
#         # print(f"move_ahead:{move_ahead} near_edge: {near_edge}, low_health: {low_health}, "
#         #       f", falling_speed: {falling_speed}")
#
#         # Combine all the features into a vector
#         # Flatten nn_input_tensor if it's not already 1D
#         features = np.array([move_ahead, alive, near_edge, low_health, falling_speed])
#         nn_input_flat = nn_input_tensor.numpy().flatten()
#     combined_features = np.concatenate([features, nn_input_flat])
#     return combined_features

def render_diagonal_coordinates(screen):
    global world
    diagonal_coords = world.get_diagonal_tile_coordinates()  # Get the coordinates as a dictionary

    # Render Top-left to Bottom-right diagonal
    for (x, y) in diagonal_coords["top_left_to_bottom_right"]:
        pygame.draw.rect(screen, (255, 0, 0), (x, y, TILE_SIZE_W, TILE_SIZE_H), 2)  # Red for TL to BR

    # Render Top-right to Bottom-left diagonal
    for (x, y) in diagonal_coords["top_right_to_bottom_left"]:
        pygame.draw.rect(screen, (0, 0, 255), (x, y, TILE_SIZE_W, TILE_SIZE_H), 2)  # Blue for TR to BL

    # Render Other diagonals
    for (x, y) in diagonal_coords["other_diagonals"]:
        pygame.draw.rect(screen, (0, 255, 0), (x, y, TILE_SIZE_W, TILE_SIZE_H), 2)  # Green for other diagonals

    # Optionally, render a filled circle for each diagonal point
    for (x, y) in diagonal_coords["top_left_to_bottom_right"]:
        pygame.draw.circle(screen, (255, 0, 0), (x + TILE_SIZE_W // 2, y + TILE_SIZE_H // 2), 5)  # Small red circle

    for (x, y) in diagonal_coords["top_right_to_bottom_left"]:
        pygame.draw.circle(screen, (0, 0, 255), (x + TILE_SIZE_W // 2, y + TILE_SIZE_H // 2), 5)  # Small blue circle

    for (x, y) in diagonal_coords["other_diagonals"]:
        pygame.draw.circle(screen, (0, 255, 0), (x + TILE_SIZE_W // 2, y + TILE_SIZE_H // 2), 5)  # Small green circle


def run_game():
    global run, start_game, world, life_line, next_life, chosen_action, action_type, world, player,enemies, health_bar, world, intro_fade, death_fade, episode, total_reward, shoot, grenade, level, moving_left, moving_right, bg_scroll, grenade_thrown, start_intro, save_data, done, epsilon

    # Initialize the extractor
    save_manager = SaveFutureLearning(MODEL_PATH, EPSILON_PATH, EPISODE_PATH)
    episode = save_manager.load_episode()
    logger = TrainingLogger()
    visualizer = TrainingVisualizer()
    extract_state = ExtractGameState(int(SCREEN_WIDTH * 0.4), int(SCREEN_HEIGHT * 0.4))
    image_ex = ImageExtractorThread(screen, extract_state)
    image_ex.start()

    if DEBUG:
        manual_play()
    else:
        # Initialize agent with CNN
        update()  # Ensure screen is rendered
        pygame.display.update()
        pygame.time.wait(500)  # Wait for 100 ms

        #AI starts
        extra_features = extract_additional_features()
        extra_features_dim = len(extra_features)
        dummy_state = image_ex.get_current_frame()
        agent = DQNAgent(dummy_state.shape, extra_features_dim, len(GameActions))
        agent.q_network, agent.target_network = save_manager.load_model(agent.q_network, agent.target_network)
        agent.epsilon = save_manager.load_epsilon() if save_manager.load_epsilon() else EPSILION
        episode = save_manager.load_episode()
        agent.update_target_network()
        reward_ai = RewardAI()
        iteration = 0
        hard_reset = False
        max_time = 0

        while run:
            clock.tick(FPS)
            world.tilescroll += get_screen_scroll()

            current_time = pygame.time.get_ticks()
            elapsed_time = (current_time - level_start_time) // 1000  # Convert to seconds
            if episode >= 10:
                max_time = 60
            else:
                max_time = 30

            if elapsed_time >= max_time or reward_ai.calculate_total_reward() < -10000:
                hard_reset = True
                done = True

            if save_data:
                current_state = image_ex.get_current_frame()
                if current_state is not None:
                    additional_features = extract_additional_features()
                    action_type, _, chosen_action, epsilon = agent.act(current_state, additional_features)
                    perform_action(GameActions(chosen_action))

                    prev_coin = player.coin
                    prev_health = player.health

                    # Update game and capture next state
                    update()

                    if not world.raycast_for_gap(player):
                        print("here")

                    next_state = image_ex.get_current_frame()
                    next_additional_features = extract_additional_features()

                    curr_coin = player.coin
                    has_gained_coin = False
                    if (curr_coin - prev_coin) > 0:
                        has_gained_coin = True

                    curr_health = player.health
                    has_gained_health = False
                    if (curr_health - prev_health) > 0:
                        has_gained_health = True


                    # Calculate reward
                    died = not player.alive
                    reached_exit = player.reached_exit()
                    # Terminal state checks
                    done = died or reached_exit or player.fell_or_hit_water()

                    reward = reward_ai.calculate_reward(chosen_action=chosen_action, player=player, has_gained_coin=has_gained_coin,has_gained_health=has_gained_health, died=died, reached_exit=reached_exit, enemy_group=enemy_group, bullet_group=bullet_group)
                    agent.save_reward(reward, chosen_action)
                    total_reward = reward_ai.calculate_total_reward()


                    print(f"Iteration: {iteration}, Type: {action_type}, Epsilon: {agent.epsilon:.4f}, "
                          f"Action: {GameActions(chosen_action).name}, Reward: {reward:.2f}, "
                          f"Total: {reward_ai.total_reward:.2f}, Health: {player.health}, died: {died}, f: {player.fell_or_hit_water()}")


                    # Store experience and train
                    agent.remember(current_state, additional_features, chosen_action, reward, next_state, next_additional_features, done)
                    # agent.update_epsilon()
                iteration += 1

            # Logging
            if (hard_reset or done) and save_data:
                # Update Log data
                total_reward = reward_ai.calculate_total_reward()
                logger.log(episode, total_reward, agent.epsilon)
                save_manager.save_model(agent.q_network, agent, episode)

                reached_exit = player.reached_exit()
                success = int(reached_exit)

                # Calculate elapsed time
                current_time = pygame.time.get_ticks()
                elapsed_time = (current_time - level_start_time) // 1000  # Convert to seconds

                # Calculate progress percentage
                max_distance = exit_x - start_x
                current_progress = player.rect.x - start_x
                progress_pct = (current_progress / max_distance) * 100 if max_distance > 0 else 0
                progress_pct = max(0, min(100, progress_pct))  # Clamp between 0-100


                # Save all relevant metrics
                visualizer.save_episode(
                    episode=episode,
                    total_reward=total_reward,
                    success=success,
                    epsilon=agent.epsilon,
                    steps=episode,
                    time_taken=elapsed_time,
                    distance_traveled=progress_pct
                )
                print(f"total_reward: {total_reward}")
                print(f"episode: {episode}")
                print(f"epsilon: {epsilon}")

                agent.replay()
                reward_ai = RewardAI()
                agent.update_epsilon()

                episode += 1
                iteration = 0
                save_data = False
                hard_reset = False
                # if episode % 5 == 0:
                #     agent.decay_epsilon()  # Decay epsilon HERE (once per episode)

            if not save_data:
                reset_game(True)

            # Event Handling
            for event in pygame.event.get():
                # quit game
                if event.type == pygame.QUIT:
                    run = False
                # Keyboard presses for player movement and actions
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        moving_left = True
                    if event.key == pygame.K_d:
                        moving_right = True
                    if event.key == pygame.K_SPACE:
                        shoot = True
                    if event.key == pygame.K_q:
                        grenade = True
                    if event.key == pygame.K_w and player.alive:
                        player.jump = True
                        jump_fx.play()
                    if event.key == pygame.K_ESCAPE:
                        run = False
                # Keyboard button released
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_a:
                        moving_left = False
                    if event.key == pygame.K_d:
                        moving_right = False
                    if event.key == pygame.K_SPACE:
                        shoot = False
                    if event.key == pygame.K_q:
                        grenade = False
                        grenade_thrown = False

            pygame.display.update()
        pygame.quit()
        visualizer.plot_progress()
        visualizer.plot_combined()

        df = visualizer.load_data()
        print("\nTraining Statistics Summary:")
        print(df.describe())





