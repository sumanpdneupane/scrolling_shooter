import pygame
from pygame import mixer

mixer.init()
pygame.init()


from src.settings import *
from src.environment.entities import ScreenFade
from src.environment.equipments import Grenade
from src.environment.world import World
from src.ai_agent.extract_state import ExtractState
from src.ai_agent.agent import DQNAgent


intro_fade = ScreenFade(1, BLACK, 4)
death_fade = ScreenFade(2, PINK, 4)

run = True
world = World()
player, health_bar = world.process_data(get_world_data())

def create_game_menu():
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

def run_game():
    global run, start_game, world, player, health_bar, intro_fade, death_fade, shoot, grenade, level, moving_left, moving_right, bg_scroll, grenade_thrown, start_intro

    new_state = ExtractState()
    dqn_agent = DQNAgent()

    while run:
        clock.tick(FPS)

        # 1. Get current state
        state = new_state.extract_state(player, enemy_group, exit_group)

        # 2. Choose Action by AI
        dqn_agent.choose_action(state)

        if start_game == False:
            # draw menu
            screen.fill(BG)
            # add buttons
            if start_button.draw(screen):
                start_game = True
                start_intro = True
            if exit_button.draw(screen):
                run = False
        else:
            # Create Game Menu
            create_game_menu()

            # Update Game
            update_game()

            # show intro
            if start_intro == True:
                if intro_fade.fade():
                    start_intro = False
                    intro_fade.fade_counter = 0

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
                if death_fade.fade():
                    if restart_button.draw(screen):
                        death_fade.fade_counter = 0
                        start_intro = True
                        bg_scroll = 0
                        world_data = reset_level()
                        world = World()
                        player, health_bar = world.process_data(world_data)

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


