import threading

import pygame
import csv
from src.utils import button

# from pygame import mixer
#
# mixer.init()
# pygame.init()
# Use a lock for synchronizing access to global variables
game_state_lock = threading.Lock()

DEBUG = False
DEBUG_SHOW_COLLISION_BOX = False
DEBUG_ENABLE_SOUND = False
SCREEN_WIDTH = 800
SCREEN_HEIGHT = int(SCREEN_WIDTH * 0.8)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Shooter')

#set framerate
clock = pygame.time.Clock()
FPS = 40

#define game variables
GRAVITY = 0.65
SCROLL_THRESH = 200
ROWS = 16
COLS = 150
TILE_SIZE = SCREEN_HEIGHT // ROWS
TILE_TYPES = 21
MAX_LEVELS = 3
GROUND_THRESHOLD = TILE_SIZE * 1.4
screen_scroll = 0
bg_scroll = 0
level = 0
start_game = False
start_intro = False


#define player action variables
moving_left = False
moving_right = False
shoot = False
grenade = False
grenade_thrown = False

# Log files
MODEL_PATH = "src/data_logs/model.pt"
EPSILON_PATH = "src/data_logs/epsilon.txt"
EPISODE_PATH = "src/data_logs/episode.txt"
TRAINING_LOG_PATH = "src/data_logs/training_log.csv"

#load music and sounds
pygame.mixer.music.load('src/assets/audio/music2.mp3')
pygame.mixer.music.set_volume(0.05 if DEBUG_ENABLE_SOUND else 0.0)
pygame.mixer.music.play(-1, 0.0, 5000)
jump_fx = pygame.mixer.Sound('src/assets/audio/jump.wav')
jump_fx.set_volume(0.03 if DEBUG_ENABLE_SOUND else 0.0)
shot_fx = pygame.mixer.Sound('src/assets/audio/shot.wav')
shot_fx.set_volume(0.04 if DEBUG_ENABLE_SOUND else 0.0)
grenade_fx = pygame.mixer.Sound('src/assets/audio/grenade.wav')
grenade_fx.set_volume(0.07 if DEBUG_ENABLE_SOUND else 0.0)


#load images
#button images
start_img = pygame.image.load('src/assets/images/start_btn.png').convert_alpha()
exit_img = pygame.image.load('src/assets/images/exit_btn.png').convert_alpha()
restart_img = pygame.image.load('src/assets/images/restart_btn.png').convert_alpha()
#background
scale = 2
sky_img = pygame.image.load('src/assets/images/Background/sky_cloud_1.png').convert_alpha()
sky_img = pygame.transform.scale(sky_img, (int(sky_img.get_width() * scale), int(sky_img.get_height() * 1.5)))
mountain_img = pygame.image.load('src/assets/images/Background/background_all.png').convert_alpha()
mountain_img = pygame.transform.scale(mountain_img, (int(mountain_img.get_width() * scale), int(mountain_img.get_height() * 1.5)))

#store tiles in a list
img_list = []
for x in range(TILE_TYPES):
	img = pygame.image.load(f'src/assets/images/Tile/{x}.png')
	img = pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE))
	img_list.append(img)
#bullet
bullet_img = pygame.image.load('src/assets/images/icons/bullet.png').convert_alpha()
bullet_img = pygame.transform.scale(bullet_img, (int(bullet_img.get_width() * 0.75), int(bullet_img.get_height() * 0.75)))

#grenade
grenade_img = pygame.image.load('src/assets/images/icons/grenade.png').convert_alpha()
#pick up boxes
health_box_img = pygame.image.load('src/assets/images/icons/health_box.png').convert_alpha()
health_box_img = pygame.transform.scale(health_box_img, (int(health_box_img.get_width() * scale), int(health_box_img.get_height() * scale)))
ammo_box_img = pygame.image.load('src/assets/images/icons/ammo_box.png').convert_alpha()
ammo_box_img = pygame.transform.scale(ammo_box_img, (int(ammo_box_img.get_width() * 1.5), int(ammo_box_img.get_height() * 1.5)))
grenade_box_img = pygame.image.load('src/assets/images/icons/grenade_box.png').convert_alpha()
grenade_box_img = pygame.transform.scale(grenade_box_img, (int(grenade_box_img.get_width() * 1.5), int(grenade_box_img.get_height() * 1.5)))

item_boxes = {
	'Health'	: health_box_img,
	'Ammo'		: ammo_box_img,
	'Grenade'	: grenade_box_img
}

def clip_image(img, size_x, size_y):
	# Clip 20% from left, right, and top — not bottom
	width, height = img.get_width(), img.get_height()
	clip_x = int(width * size_x)
	clip_y = int(height * size_y)
	new_width = width - 2 * clip_x
	new_height = height - clip_y  # don't remove from bottom

	# Crop the image
	img = img.subsurface((clip_x, clip_y, new_width, new_height)).copy()
	return img


#define colours
BG = (144, 201, 120)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
PINK = (235, 65, 54)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

#define font
font = pygame.font.SysFont('Futura', 18)

#create sprite groups
enemy_group = pygame.sprite.Group()
bullet_group = pygame.sprite.Group()
grenade_group = pygame.sprite.Group()
explosion_group = pygame.sprite.Group()
item_box_group = pygame.sprite.Group()
decoration_group = pygame.sprite.Group()
water_group = pygame.sprite.Group()
exit_group = pygame.sprite.Group()


#create buttons
start_button = button.Button(SCREEN_WIDTH // 2 - 130, SCREEN_HEIGHT // 2 - 150, start_img, 1)
exit_button = button.Button(SCREEN_WIDTH // 2 - 110, SCREEN_HEIGHT // 2 + 50, exit_img, 1)
restart_button = button.Button(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 50, restart_img, 2)

def draw_text(text, font, text_col, x, y):
	img = font.render(text, True, text_col)
	screen.blit(img, (x, y))


def draw_bg():
	screen.fill(BG)
	# width = sky_img.get_width()
	width = mountain_img.get_width()
	for x in range(5):
		screen.blit(sky_img, ((x * width) - bg_scroll * 0.5, 0))
		screen.blit(mountain_img, ((x * width) - bg_scroll * 0.6, SCREEN_HEIGHT - mountain_img.get_height()))
		# screen.blit(pine1_img, ((x * width) - bg_scroll * 0.7, SCREEN_HEIGHT - pine1_img.get_height() - 150))
		# screen.blit(pine2_img, ((x * width) - bg_scroll * 0.8, SCREEN_HEIGHT - pine2_img.get_height()))


#function to reset level
def reset_level():
	enemy_group.empty()
	bullet_group.empty()
	grenade_group.empty()
	explosion_group.empty()
	item_box_group.empty()
	decoration_group.empty()
	water_group.empty()
	exit_group.empty()

	#create empty tile list
	return get_world_data(level)

_screen_scroll = 0

def get_screen_scroll():
    global _screen_scroll
    return _screen_scroll

def set_screen_scroll(screen_scroll):
    global _screen_scroll
    _screen_scroll = screen_scroll

def get_world_data(level):
	# create empty tile list
	world_data = []
	for row in range(ROWS):
		r = [-1] * COLS
		world_data.append(r)
	# load in level data and create world
	with open(f'src/assets/level_data/level{level}_data.csv', newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for x, rows in enumerate(reader):
			for y, tile in enumerate(rows):
				world_data[x][y] = int(tile)
	return world_data