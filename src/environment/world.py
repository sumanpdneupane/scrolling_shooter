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
    					player = Soldier('player', x * TILE_SIZE, y * TILE_SIZE, 1.5, 5, 20, 5)
    					health_bar = HealthBar(10, 10, player.health, player.health)
    				elif tile == 16:#create enemies
    					enemy = Soldier('enemy', x * TILE_SIZE, y * TILE_SIZE, 1.125, 2, 20, 0)
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
