import time

import numpy as np
import pygame
from pygame import mixer

mixer.init()
pygame.init()

from src.settings import *
from src.environment.entities import ScreenFade
from src.environment.world import World
from src.environment.equipments import Grenade
from src.ai_agent.agent_state_and_action import ExtractGameState, GameActions
from src.ai_agent.agent import DQNAgent
from src.ai_agent.reward import RewardAI
from src.ai_agent.save_model_data import SaveFutureLearning
from src.utils.logger import TrainingLogger, GraphLogger

intro_fade = ScreenFade(1, BLACK, 4)
death_fade = ScreenFade(2, PINK, 4)

run = True
world = World()
player, health_bar = world.process_data(get_world_data())

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


def update_game_menu():
    global world, player, health_bar, grenade, level
    # update background
    draw_bg()
    # draw world map
    world.draw()
    # show player health
    health_bar.draw(player.health)
    # show ammo
    draw_text('AMMO: ', font, WHITE, 10, 35)
    for x in range(player.ammo):
        screen.blit(bullet_img, (90 + (x * 10), 40))
    # show grenades
    draw_text('GRENADES: ', font, WHITE, 10, 60)
    for x in range(player.grenades):
        screen.blit(grenade_img, (135 + (x * 15), 60))


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
    global run, start_game, start_intro, world, player, health_bar, intro_fade, death_fade, shoot, grenade, level, moving_left, moving_right, bg_scroll, grenade_thrown

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
            world_data = reset_level()
            if level <= MAX_LEVELS:
                world = World()
                player, health_bar = world.process_data(world_data)
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
        show_intro_fade()

        # update player actions
        update_player_action()


def reset_game(from_agent_click=False):
    global run, done, save_data, start_game, world, player, health_bar, intro_fade, death_fade, shoot, grenade, level, moving_left, moving_right, bg_scroll, grenade_thrown, start_intro
    if death_fade.fade():
        if restart_button.draw(screen) or from_agent_click:
            save_data = True
            death_fade.fade_counter = 0
            start_intro = True
            bg_scroll = 0
            world_data = reset_level()
            world = World()
            player, health_bar = world.process_data(world_data)
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
        if player.alive and not player.in_air:
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

def run_game():
    global run, start_game, world, player, health_bar, intro_fade, death_fade, shoot, grenade, level, moving_left, moving_right, bg_scroll, grenade_thrown, start_intro, save_data, done
    extract_state = ExtractGameState()

    # while run:
    #     clock.tick(FPS)
    #     update()
    #
    #     state_dict, state = extract_state.extract_state(player, world, enemy_group, exit_group)
    #
    #     # Event Handling
    #     for event in pygame.event.get():
    #         # quit game
    #         if event.type == pygame.QUIT:
    #             run = False
    #         # Only process input if the agent is not controlling
    #         # if not save_data:
    #         # keyboard presses
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_a:
    #                 moving_left = True
    #             if event.key == pygame.K_d:
    #                 moving_right = True
    #             if event.key == pygame.K_SPACE:
    #                 shoot = True
    #             if event.key == pygame.K_q:
    #                 grenade = True
    #             if event.key == pygame.K_w and player.alive:
    #                 player.jump = True
    #                 jump_fx.play()
    #             if event.key == pygame.K_ESCAPE:
    #                 run = False
    #         # keyboard button released
    #         if event.type == pygame.KEYUP:
    #             if event.key == pygame.K_a:
    #                 moving_left = False
    #             if event.key == pygame.K_d:
    #                 moving_right = False
    #             if event.key == pygame.K_SPACE:
    #                 shoot = False
    #             if event.key == pygame.K_q:
    #                 grenade = False
    #                 grenade_thrown = False
    #
    #     pygame.display.update()
    # pygame.quit()

    save_manager = SaveFutureLearning(MODEL_PATH, EPSILON_PATH, EPISODE_PATH)
    episode = save_manager.load_episode()
    extract_state = ExtractGameState()
    dummy_state_dict, dummy_state = extract_state.extract_state(player, world, enemy_group, exit_group)
    state_dim = dummy_state.shape[0]
    action_dim = len(GameActions)
    agent = DQNAgent(state_dim, action_dim)
    reward_ai = RewardAI()
    reward_ai.reset_total_reward()
    logger = TrainingLogger()

    # load manager
    save_manager.load_model(agent.q_network, agent.target_network, agent)
    agent.update_target_network()  # Sync target network with Q-network
    iteration = 0
    while run:
        clock.tick(FPS)

        if save_data:
            player.prev_health = player.health

            # Reset temporary counters BEFORE action
            player.bullets_hit_this_frame = 0
            player.moved_forward = False

            # Old State
            state_dict, state = extract_state.extract_state(player, world, enemy_group, exit_group)
            print(state_dict)
            action_type, rand_values, chosen_action = agent.act(state_dict, state, episode)
            perform_action(GameActions(chosen_action))

            # Update Game - this will modify the state
            update()

            # --- Check post-update values AFTER game updates ---
            died = not player.alive
            reached_exit = player.reached_exit()

            # New State
            next_state = extract_state.extract_state(player, world, enemy_group, exit_group)
            reward = reward_ai.calculate_reward(state_dict, state, chosen_action, died)

            # Terminal state checks
            done = not player.alive or reached_exit

            # print(f"Iteration: {iteration}, Type: {action_type}, Epsilon: {agent.epsilon:.4f}, Random: {rand_values}, "
            #       f"Action: {GameActions(chosen_action).name}, Reward: {reward:.2f}, "
            #       f"Total: {reward_ai.total_reward:.2f}, Health: {player.health}")

            # Add to reward calculation
            if DEBUG:
                pygame.draw.line(screen, RED, player.rect.center,
                                 (player.rect.centerx + 50 * player.direction,
                                  player.rect.centery), 3)

            # Rember
            agent.remember(state, chosen_action, reward, next_state, done)
            agent.replay(episode)
            iteration += 1


        # Logging
        if done and save_data:
            # Update Log data
            total_reward = reward_ai.calculate_total_reward()
            logger.log(episode, total_reward, agent.epsilon)
            save_manager.save_model(agent.q_network, agent, episode)
            # save_manager.load_model(agent.q_network, agent.target_network, agent)

            print(f"episode: {episode}")
            episode += 1
            iteration = 0
            save_data = False
            reward_ai.reset_total_reward()
            agent.end_episode()  # Decay epsilon HERE (once per episode)
        if not save_data:
            reset_game(True)

        # Event Handling
        for event in pygame.event.get():
            # quit game
            if event.type == pygame.QUIT:
                run = False
            # Only process input if the agent is not controlling
            # if not save_data:
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
