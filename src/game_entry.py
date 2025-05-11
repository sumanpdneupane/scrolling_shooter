import time
import pygame
from pygame import mixer
from src.graph.plot import TrainingVisualizer

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
    global world, player, health_bar, grenade, level, start_x, exit_x
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

    # Calculate elapsed time
    current_time = pygame.time.get_ticks()
    elapsed_time = (current_time - level_start_time) // 1000  # Convert to seconds
    minutes = elapsed_time // 60
    seconds = elapsed_time % 60
    time_str = f"{minutes:02}:{seconds:02}"  # Format as MM:SS

    # Show game timer in top-right corner
    # Show game timer and level in top-right corner
    draw_text(f'LEVEL: {level} / {MAX_LEVELS}', font, WHITE, SCREEN_WIDTH - 150, 10)
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
            world_data = get_world_data()
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
        show_intro_fade()

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

    if DEBUG:
        while run:
            clock.tick(FPS)
            update()

            state_dict, state = extract_state.extract_state(player, world, enemy_group, exit_group)

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
    else:
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
        # Initialize visualizer with a filename
        visualizer = TrainingVisualizer()

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
                prev_ammo = player.ammo
                prev_grenades = player.grenades
                prev_enemy_count = len(enemy_group)  # Track enemies before action


                # Old State
                state_dict, state = extract_state.extract_state(player, world, enemy_group, exit_group)
                print(state_dict)
                action_type, rand_values, chosen_action = agent.act(state)

                # Detect if a shot was fired
                shot_fired = (GameActions(chosen_action) == GameActions.SHOOT)

                perform_action(GameActions(chosen_action))

                # Update Game - this will modify the state
                update()

                # --- Check post-update values AFTER game updates ---
                died = not player.alive
                reached_exit = player.reached_exit()
                health_gained = player.health > player.prev_health
                ammo_gained = player.ammo > prev_ammo
                grenades_gained = player.grenades > prev_grenades

                # Detect if any enemy died
                enemy_killed = len(enemy_group) < prev_enemy_count

                # New State
                next_state = extract_state.extract_state(player, world, enemy_group, exit_group)
                reward = reward_ai.calculate_reward(state_dict, state, chosen_action, died, enemy_killed, shot_fired, health_gained, ammo_gained, grenades_gained, reached_exit)

                # Terminal state checks
                done = not player.alive or reached_exit

                # print(f"Iteration: {iteration}, Type: {action_type}, Epsilon: {agent.epsilon:.4f}, Random: {rand_values}, "
                #       f"Action: {GameActions(chosen_action).name}, Reward: {reward:.2f}, "
                #       f"Total: {reward_ai.total_reward:.2f}, Health: {player.health}")

                # Add to reward calculation
                if DEBUG_SHOW_COLLISION_BOX:
                    pygame.draw.line(screen, RED, player.rect.center,
                                     (player.rect.centerx + 50 * player.direction,
                                      player.rect.centery), 3)

                # Rember
                agent.remember(state, chosen_action, reward, next_state, done)
                agent.replay(episode)
                iteration += 1

            # Calculate elapsed time
            current_time = pygame.time.get_ticks()
            elapsed_time = (current_time - level_start_time) // 1000  # Convert to seconds
            if elapsed_time > 60:
                done = True
                save_data = True

            # Logging
            if done and save_data:
                # Update Log data
                total_reward = reward_ai.calculate_total_reward()
                logger.log(episode, total_reward, agent.epsilon)
                save_manager.save_model(agent.q_network, agent, episode)
                # save_manager.load_model(agent.q_network, agent.target_network, agent)

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


                print("total_reward", total_reward)
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

                print(f"episode: {episode}")
                episode += 1
                iteration = 0
                save_data = False
                reward_ai.reset_total_reward()
                agent.decay_epsilon()  # Decay epsilon HERE (once per episode)

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

        try:
            visualizer.plot_progress()
            visualizer.plot_combined()


            df = visualizer.load_data()
            print("\nTraining Statistics Summary:")
            print(df.describe())

            # # Example filtered plot
            # if len(df) > 2000:
            #     filtered_df = df[df['episode'].between(1000, 2000)]
            #     plt.figure(figsize=(12, 6))
            #     sns.lineplot(data=filtered_df, x='episode', y='total_reward')
            #     plt.title("Reward Progression (Episodes 1000-2000)")
            #     plt.show()
        except Exception as e:
            print(f"Could not generate final plots: {e}")
