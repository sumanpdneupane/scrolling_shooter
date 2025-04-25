from agent_state_and_action import GameActions

class DQNAgent():
    def choose_action(self, state):
        print(state)
        # Example AI logic:
        return GameActions.Jump
        # if state["ammo"] > 0 and state["nearest_enemy_dx"] < 0.5:
        #     return "shoot"
        # elif state["health_percentage"] < 0.3:
        #     return "move_left"  # Retreat
        # elif state["near_exit"] == 1:
        #     return "move_right"  # Head towards exit
        # elif state["player_in_air"] == 0:
        #     return "jump"  # Jump if not in air
        # else:
        #     return "do_nothing"  # Default action