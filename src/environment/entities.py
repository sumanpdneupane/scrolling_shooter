from src.settings import *

class ItemBox(pygame.sprite.Sprite):
    def __init__(self, item_type, x, y, player_reference=None):
        pygame.sprite.Sprite.__init__(self)
        self.item_type = item_type
        self.image = item_boxes[self.item_type]
        self.rect = self.image.get_rect()
        self.rect.midtop = (x + TILE_SIZE // 2, y + (TILE_SIZE - self.image.get_height()))
        self.player = player_reference

    def update(self, player=None):
        # Update player reference if provided
        if player is not None:
            self.player = player

        if self.player is None:
            return  # Can't function without player reference

        # Scroll with screen
        self.rect.x += get_screen_scroll()

        # Check collision with player
        if pygame.sprite.collide_rect(self, self.player):
            # check what kind of box it was
            if self.item_type == 'Health':
                player.health += 25
                if player.health > player.max_health:
                    player.health = player.max_health
            elif self.item_type == 'Ammo':
                player.ammo += 15
            elif self.item_type == 'Grenade':
                player.grenades += 3
            # delete the item box
            self.kill()


class Decoration(pygame.sprite.Sprite):
    def __init__(self, img, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = img
        self.rect = self.image.get_rect()
        self.rect.midtop = (x + TILE_SIZE // 2, y + (TILE_SIZE - self.image.get_height()))


    def update(self):
        global screen_scroll
        self.rect.x += get_screen_scroll()


class Water(pygame.sprite.Sprite):
    def __init__(self, img, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = img
        self.rect = self.image.get_rect()
        self.rect.midtop = (x + TILE_SIZE // 2, y + (TILE_SIZE - self.image.get_height()))


    def update(self):
        global screen_scroll
        self.rect.x += get_screen_scroll()


class Exit(pygame.sprite.Sprite):
    def __init__(self, img, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = img
        self.rect = self.image.get_rect()
        self.rect.midtop = (x + TILE_SIZE // 2, y + (TILE_SIZE - self.image.get_height()))


    def update(self):
        global screen_scroll
        self.rect.x += get_screen_scroll()




class HealthBar():
    def __init__(self, x, y, health, max_health):
        self.x = x
        self.y = y
        self.health = health
        self.max_health = max_health


    def draw(self, health):
        #update with new health
        self.health = health
        #calculate health ratio
        ratio = self.health / self.max_health

        # Reduced height parameters
        border_height = 12  # Original: 24
        inner_height = 10  # Original: 20

        # Draw border (now 14px tall instead of 24)
        pygame.draw.rect(screen, WHITE, (self.x - 1, self.y - 1, 168, border_height))
        # Draw background (now 10px tall instead of 20)
        pygame.draw.rect(screen, RED, (self.x, self.y, 166, inner_height))
        # Draw health (now 10px tall instead of 20)
        pygame.draw.rect(screen, GREEN, (self.x, self.y, 166 * ratio, inner_height))

        # pygame.draw.rect(screen, BLACK, (self.x - 2, self.y - 2, 154, 20))
        # pygame.draw.rect(screen, RED, (self.x, self.y, 150, 20))
        # pygame.draw.rect(screen, GREEN, (self.x, self.y, 150 * ratio, 20))



class ScreenFade():
    def __init__(self, direction, colour, speed):
        self.direction = direction
        self.colour = colour
        self.speed = speed
        self.fade_counter = 0

    def fade(self):
        fade_complete = False
        self.fade_counter += self.speed
        if self.direction == 1:#whole screen fade
            pygame.draw.rect(screen, self.colour, (0 - self.fade_counter, 0, SCREEN_WIDTH // 2, SCREEN_HEIGHT))
            pygame.draw.rect(screen, self.colour, (SCREEN_WIDTH // 2 + self.fade_counter, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.draw.rect(screen, self.colour, (0, 0 - self.fade_counter, SCREEN_WIDTH, SCREEN_HEIGHT // 2))
            pygame.draw.rect(screen, self.colour, (0, SCREEN_HEIGHT // 2 +self.fade_counter, SCREEN_WIDTH, SCREEN_HEIGHT))
        if self.direction == 2:#vertical screen fade down
            pygame.draw.rect(screen, self.colour, (0, 0, SCREEN_WIDTH, 0 + self.fade_counter))
        if self.fade_counter >= SCREEN_WIDTH:
            fade_complete = True


        return fade_complete



