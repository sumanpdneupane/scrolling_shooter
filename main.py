from src.game_entry import *

run_game()


# import threading
# import time
#
# # Global Variables
# player_health = 100
# player_score = 0
# lock = threading.Lock()
#
# # Main Thread Function (Simulates UI and Game Logic)
# def main_thread():
#     global player_health, player_score
#     while player_health > 0:
#         with lock:
#             player_score += 1  # Increment player score every second
#             print(f"Main Thread - Player Score: {player_score}")
#         time.sleep(1)
#
# # Game Thread Function (Simulates Game World Updates)
# def game_thread():
#     global player_health, player_score
#     while player_health > 0:
#         with lock:
#             player_health -= 5  # Decrease health over time
#             print(f"Game Thread - Player Health: {player_health}")
#         time.sleep(2)
#
# # Create and Start Threads
# main_thread_obj = threading.Thread(target=main_thread)
# game_thread_obj = threading.Thread(target=game_thread)
#
# main_thread_obj.start()
# game_thread_obj.start()
#
# # Wait for both threads to finish
# main_thread_obj.join()
# game_thread_obj.join()
#
# print("Game Over!")
