import enum
import numpy as np
import pygame
import threading
import time


class GameActions(enum.IntEnum):
    MOVE_RIGHT = 0
    MOVE_LEFT = 1
    STOP = 2
    JUMP = 3
    SHOOT = 4
    GRENADE = 5




class ExtractGameState:
    def __init__(self, width=84, height=84):
        self.width = width
        self.height = height

    # def extract_image(self, screen):
    #     """Efficiently captures and preprocesses the current game screen."""
    #     # Capture screen as an array (width, height, 3)
    #     img_array = pygame.surfarray.pixels3d(screen)
    #
    #     # Transpose to (height, width, 3)
    #     img_array = np.transpose(img_array, (1, 0, 2))
    #
    #     # Convert to grayscale using NumPy (faster than OpenCV for this step)
    #     gray_img = np.dot(img_array[..., :3], [0.2989, 0.587, 0.114]).astype(np.uint8)
    #
    #     # Resize to 84x84 (or specified size)
    #     resized_img = cv2.resize(gray_img, (self.width, self.height), interpolation=cv2.INTER_AREA)
    #
    #     # Normalize and add channel dimension (84, 84, 1)
    #     final_img = np.expand_dims(resized_img, axis=-1)
    #     return final_img

    def extract_image(self, screen, save_path= None):
        img_array = pygame.surfarray.pixels3d(screen)
        img_array = np.transpose(img_array, (1, 0, 2))
        gray_img = np.dot(img_array[..., :3], [0.2989, 0.587, 0.114]).astype(np.uint8)
        resized_img = gray_img[::int(gray_img.shape[0] / self.height), ::int(gray_img.shape[1] / self.width)]
        processed_img = np.expand_dims(resized_img, axis=-1)
        return processed_img


class ImageExtractorThread(threading.Thread):
    def __init__(self, screen, extract_state, region=None):
        super().__init__()
        self.screen = screen
        self.extract_state = extract_state
        self.current_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.region = region
        self.iteration = 0

    def run(self):
        while self.running:
            # Capture the relevant screen region only
            if self.region:
                rect = pygame.Rect(self.region)
                screen_copy = self.screen.subsurface(rect).copy()
            else:
                screen_copy = self.screen.copy()

            # Extract the game state from the screen copy
            frame = self.extract_state.extract_image(screen_copy)

            # # Create the directory if it doesn't exist
            # save_directory = "extract_image"
            # os.makedirs(save_directory, exist_ok=True)
            #
            # # Save the image
            # image_path = os.path.join(save_directory, f"frame_{self.iteration}.png")
            # cv2.imwrite(image_path, frame)

            self.iteration += 1
            with self.lock:
                self.current_frame = frame

            # Update the current frame
            with self.lock:
                self.current_frame = frame

            # Reduce CPU usage
            time.sleep(0.00001)

    def get_current_frame(self):
        with self.lock:
            return self.current_frame

    def stop(self):
        self.running = False

