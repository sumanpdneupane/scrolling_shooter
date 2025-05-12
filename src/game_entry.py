import os
import time
from collections import deque

import cv2
import pygame
from matplotlib import pyplot as plt
from pygame import mixer
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

intro_fade = ScreenFade(1, BLACK, 4)
death_fade = ScreenFade(2, PINK, 4)

run = True
world = World()
player, health_bar = world.process_data(get_world_data(level))

save_data = True
done = False
game_start_time = time.time()


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

    draw_text(f'Episode: {episode}', font, WHITE, SCREEN_WIDTH - 480, 20)
    draw_text(f'Total Reward:{total_reward:.2f}', font, WHITE, SCREEN_WIDTH - 480, 40)
    draw_text(f'Epsilon:{epsilon:.2f}', font, WHITE, SCREEN_WIDTH - 480, 60)
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
            level += 1
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
        if restart_button.draw(screen) or from_agent_click:
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
    global moving_left, moving_right, shoot, grenade, grenade_thrown

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

def manual_play():
    global run, start_game, world, player, epsilon, health_bar, intro_fade, death_fade, shoot, grenade, level, moving_left, moving_right, bg_scroll, grenade_thrown, start_intro, save_data, done
    # extract_state = ExtractGameState()
    iteration = 0
    while run:
        clock.tick(FPS)
        # current_state = extract_state.extract_image(screen)
        # # Create the directory if it doesn't exist
        # save_directory = "saved_images"
        # os.makedirs(save_directory, exist_ok=True)
        #
        # # Save the image
        # image_path = os.path.join(save_directory, f"game_image{iteration}.png")
        # cv2.imwrite(image_path, current_state)
        #
        # print(f"Image saved to: {image_path}")
        update()
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

def run_game():
    global run, start_game, world, life_line, next_life, chosen_action, action_type, player, health_bar, intro_fade, death_fade, episode, total_reward, shoot, grenade, level, moving_left, moving_right, bg_scroll, grenade_thrown, start_intro, save_data, done, epsilon

    # Initialize the extractor
    save_manager = SaveFutureLearning(MODEL_PATH, EPSILON_PATH, EPISODE_PATH)
    episode = save_manager.load_episode()
    logger = TrainingLogger()
    visualizer = TrainingVisualizer()
    extract_state = ExtractGameState(int(SCREEN_WIDTH * 0.2), int(SCREEN_HEIGHT * 0.2))
    image_ex = ImageExtractorThread(screen, extract_state)
    image_ex.start()

    if DEBUG:
        manual_play()
    else:
        # Initialize agent with CNN
        update()  # Ensure screen is rendered
        pygame.display.update()
        pygame.time.wait(100)  # Wait for 100 ms

        dummy_state = image_ex.get_current_frame()
        # state_shape = (dummy_state.shape[0], dummy_state.shape[1])  # (84, 84) for example Use 2D state shape (height, width)
        agent = DQNAgent(dummy_state.shape, len(GameActions))
        agent.model, agent.target_model = save_manager.load_model(agent.model, agent.target_model)
        agent.epsilon = save_manager.load_epsilon()
        agent.update_target_network()
        reward_ai = RewardAI()
        iteration = 0

        # player_x, player_y = player.rect.center
        # player_spon_point = {
        #     "x": player_x,
        #     "y": player_y
        # }
        # print(f"Player Position: ({player_spon_point})")
        hard_reset = False
        while run:
            clock.tick(FPS)

            if reward_ai.calculate_total_reward() < -8000:
                hard_reset = True


            if save_data:
                # Capture previous state
                prev_health = player.health
                prev_ammo = player.ammo
                prev_grenades = player.grenades
                prev_enemy_count = len(enemy_group)

                current_state = image_ex.get_current_frame()
                if current_state is not None:
                    action_type, _, chosen_action, epsilon = agent.act(current_state)
                    perform_action(GameActions(chosen_action))

                    # Update game and capture next state
                    update()

                    next_state = image_ex.get_current_frame()

                    # Calculate reward
                    current_enemy_count = len(enemy_group)
                    enemy_killed = current_enemy_count < prev_enemy_count
                    died = not player.alive
                    reached_exit = False #player.reached_exit()
                    shot_fired = (GameActions(chosen_action) == GameActions.SHOOT)
                    reward = reward_ai.calculate_reward(chosen_action, player, enemy_killed, died, reached_exit, shot_fired, prev_health, prev_ammo, prev_grenades, world)
                    total_reward = reward_ai.calculate_total_reward()

                    # Terminal state checks
                    done = died or reached_exit

                    print(f"Iteration: {iteration}, Type: {action_type}, Epsilon: {agent.epsilon:.4f}, "
                          f"Action: {GameActions(chosen_action).name}, Reward: {reward:.2f}, "
                          f"Total: {reward_ai.total_reward:.2f}, Health: {player.health}")


                    # Store experience and train
                    agent.remember(current_state, chosen_action, reward, next_state, done)
                    agent.replay(32)  # Adjust batch size as needed
                    # agent.decay_epsilon()
                iteration += 1

            # Logging
            if (hard_reset or done) and save_data:
                # Update Log data
                total_reward = reward_ai.calculate_total_reward()
                logger.log(episode, total_reward, agent.epsilon)
                save_manager.save_model(agent.model, agent, episode)

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

                episode += 1
                iteration = 0
                save_data = False
                hard_reset = False

                reward_ai.reset_total_reward()
                if episode % 5 == 0:
                    agent.decay_epsilon()  # Decay epsilon HERE (once per episode)

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

        # Visualize the model structure
        agent.visualize_model()
        # Visualize the first convolutional layer
        agent.visualize_weights(layer_index=0)
        # You can also visualize the second and third layers
        agent.visualize_weights(layer_index=1)
        agent.visualize_weights(layer_index=2)


# def run_game():
#     global run, start_game, world, life_line, next_life, player, health_bar, intro_fade, death_fade, episode, total_reward, shoot, grenade, level, moving_left, moving_right, bg_scroll, grenade_thrown, start_intro, save_data, done
#
#     # Initialize the extractor
#     save_manager = SaveFutureLearning(MODEL_PATH, EPSILON_PATH, EPISODE_PATH)
#     episode = save_manager.load_episode()
#     logger = TrainingLogger()
#     visualizer = TrainingVisualizer()
#     extract_state = ExtractGameState(int(SCREEN_WIDTH * 0.2), int(SCREEN_HEIGHT * 0.2))
#     image_ex = ImageExtractorThread(screen, extract_state)
#     image_ex.start()
#
#     # Initialize the agent
#     update()
#     pygame.display.update()
#     pygame.time.wait(100)
#     dummy_state = image_ex.get_current_frame()
#     agent = DQNAgent(dummy_state.shape, len(GameActions))
#     agent.model, agent.target_model = save_manager.load_model(agent.model, agent.target_model)
#     agent.epsilon = save_manager.load_epsilon()
#     agent.update_target_network()
#     reward_ai = RewardAI()
#     iteration = 0
#     player_spon_point = {"x": player.rect.centerx, "y": player.rect.centery}
#     life_line = 3
#
#     while run:
#         clock.tick(FPS)
#
#         if save_data:
#             # Capture previous state
#             prev_health = player.health
#             prev_ammo = player.ammo
#             prev_grenades = player.grenades
#             prev_enemy_count = len(enemy_group)
#             current_state = image_ex.get_current_frame()
#
#             if current_state is not None:
#                 action_type, _, chosen_action, epsilon = agent.act(current_state)
#                 perform_action(GameActions(chosen_action))
#                 update()
#                 next_state = image_ex.get_current_frame()
#
#                 # Calculate reward
#                 current_enemy_count = len(enemy_group)
#                 enemy_killed = current_enemy_count < prev_enemy_count
#                 died = not player.alive
#                 shot_fired = (GameActions(chosen_action) == GameActions.SHOOT)
#                 reward = reward_ai.calculate_reward(chosen_action, player, enemy_killed, died, False, shot_fired,
#                                                     prev_health, prev_ammo, prev_grenades)
#                 total_reward = reward_ai.calculate_total_reward()
#                 done = died or player.reached_exit()
#
#                 # Store experience and train
#                 agent.remember(current_state, chosen_action, reward, next_state, done)
#                 agent.replay(32)
#
#                 # Log training metrics at episode end
#                 if done:
#                     logger.log(episode, total_reward, agent.epsilon)
#                     visualizer.save_episode(
#                         episode=episode,
#                         total_reward=total_reward,
#                         success=int(player.reached_exit()),
#                         epsilon=agent.epsilon,
#                         steps=iteration,
#                         time_taken=(pygame.time.get_ticks() - level_start_time) // 1000,
#                         distance_traveled=player.rect.centerx - player_spon_point["x"]
#                     )
#                     save_manager.save_model(agent.model, agent, episode)
#                     agent.decay_epsilon()
#                     episode += 1
#                     save_data = False
#                     reward_ai.reset_total_reward()
#                     iteration = 0
#                     print(f"Episode {episode} completed.")
#
#             iteration += 1
#
#         # Handle player life and reset logic
#         if not save_data:
#             life_line -= 1
#             if life_line > 0:
#                 player.rect.center = (player_spon_point["x"], player_spon_point["y"])
#                 player.health = player.max_health
#                 print(f"Respawning player. Life lines left: {life_line}")
#             else:
#                 life_line = 3
#                 reward_ai.reset_total_reward()
#                 reset_game(True)
#                 print("Out of life lines. Starting a new episode.")
#
#         # Event Handling
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 run = False
#             if event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_a:
#                     moving_left = True
#                 if event.key == pygame.K_d:
#                     moving_right = True
#                 if event.key == pygame.K_SPACE:
#                     shoot = True
#                 if event.key == pygame.K_q:
#                     grenade = True
#                 if event.key == pygame.K_w and player.alive:
#                     player.jump = True
#                     jump_fx.play()
#                 if event.key == pygame.K_ESCAPE:
#                     run = False
#             if event.type == pygame.KEYUP:
#                 if event.key == pygame.K_a:
#                     moving_left = False
#                 if event.key == pygame.K_d:
#                     moving_right = False
#                 if event.key == pygame.K_SPACE:
#                     shoot = False
#                 if event.key == pygame.K_q:
#                     grenade = False
#                     grenade_thrown = False
#
#         pygame.display.update()
#
#     pygame.quit()
#     visualizer.plot_progress()
#     visualizer.plot_combined()
#     df = visualizer.load_data()
#     print("\nTraining Statistics Summary:")
#     print(df.describe())
