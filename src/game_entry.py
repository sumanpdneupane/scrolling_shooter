import time

import pygame
from pygame import mixer

mixer.init()
pygame.init()


from src.settings import *
from src.environment.entities import ScreenFade
from src.environment.equipments import Grenade
from src.environment.world import World
from src.ai_agent.agent_state_and_action import ExtractGameState, GameActions
from src.ai_agent.agent import DQNAgent, RewardAI
from src.ai_agent.save_model_data import SaveFutureLearning
from src.utils.logger import TrainingLogger

intro_fade = ScreenFade(1, BLACK, 4)
death_fade = ScreenFade(2, PINK, 4)

run = True
world = World()
player, health_bar = world.process_data(get_world_data())

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

save_data = True

def reset_game(from_agent_click= False):
    global run, save_data, start_game, world, player, health_bar, intro_fade, death_fade, shoot, grenade, level, moving_left, moving_right, bg_scroll, grenade_thrown, start_intro
    if death_fade.fade():
        if restart_button.draw(screen) or from_agent_click:
            save_data = True
            death_fade.fade_counter = 0
            start_intro = True
            bg_scroll = 0
            world_data = reset_level()
            world = World()
            player, health_bar = world.process_data(world_data)

def perform_action(action):
    # Simulate key press actions
    keys = {
        GameActions.MoveLeft: pygame.K_a,
        GameActions.MoveRight: pygame.K_d,
        GameActions.Jump: pygame.K_w,
        GameActions.Shoot: pygame.K_SPACE,
        GameActions.Grenade: pygame.K_q
    }
    if action in keys:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=keys[action]))

def run_game():
    global run, start_game, world, player, health_bar, intro_fade, death_fade, shoot, grenade, level, moving_left, moving_right, bg_scroll, grenade_thrown, start_intro, save_data

    save_manager = SaveFutureLearning(MODEL_PATH, EPSILON_PATH, EPISODE_PATH)
    episode = save_manager.load_episode()
    extract_state = ExtractGameState()
    dummy_state = extract_state.extract_state(player, world, enemy_group, exit_group)
    state_dim = dummy_state.shape[0]
    action_dim = len(GameActions)
    agent = DQNAgent(state_dim, action_dim)
    reward_ai = RewardAI()
    reward_ai.reset_total_reward()
    logger = TrainingLogger()

    # load manager
    save_manager.load_model(agent.q_network, agent.target_network, agent)

    while run:
        clock.tick(FPS)

        done = False
        if save_data:
            # --- Check if any before updates  ---
            prev_enemy_count = len(enemy_group)
            prev_ammo = player.ammo
            prev_grenades = player.grenades
            prev_health = player.health

            # Old State
            state = extract_state.extract_state(player, world, enemy_group, exit_group)
            action = agent.act(state)
            perform_action(GameActions(action))

            # Update Game
            update()

            # --- Check if any after updates ---
            post_enemy_count = len(enemy_group)
            post_ammo = player.ammo
            post_grenades = player.grenades
            killed_enemy = post_enemy_count < prev_enemy_count
            fired_bullet = post_ammo < prev_ammo
            threw_grenade = post_grenades < prev_grenades
            bullet_hit_enemy = player.bullet_hit_enemy()
            fell_or_hit_water = player.fell_or_hit_water()
            reached_exit = player.reached_exit()
            walked_forward = player.walked_forward()

            # New State
            next_state = extract_state.extract_state(player, world, enemy_group, exit_group)
            reward = reward_ai.calculate_reward(prev_health, player.health, killed_enemy, fired_bullet, bullet_hit_enemy, threw_grenade, fell_or_hit_water, reached_exit, walked_forward)
            print(f"reward: {reward}, total_reward:{reward_ai.calculate_total_reward()}")
            done = player.health <= 0

            # Rember
            agent.remember(state, action, reward, next_state, done)
            agent.replay(episode)
            if pygame.time.get_ticks() % 1000 < 20:
                agent.update_target_network()

        # Logging
        if done and save_data:
            total_reward = reward_ai.calculate_total_reward()
            logger.log(episode, total_reward, agent.epsilon)
            save_manager.save_model(agent.q_network, agent)
            save_manager.load_model(agent.q_network, agent.target_network, agent)

            print(f"episode: {episode}")
            episode += 1
            save_data = False
            reward_ai.reset_total_reward()
            # print(f"Episode {episode} ended. Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
        if not save_data:
            reset_game(True)

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
