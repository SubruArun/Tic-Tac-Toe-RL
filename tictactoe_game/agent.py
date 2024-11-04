import random

# Utils
from utils import get_value


class QLearningAgent:
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.possible_actions = list(range(num_actions))
        self.q_table: dict[str, dict[int, float]] = {}
        self.q_table_update_count: dict[str, int] = {}
        self.gamma = 0.9  # discount factor
        self.epsilon = 0.5  # exploration probability
        self.alpha = 0.1  # learning rate

    def agent_action(self, observation: str, game, num_games) -> int:
        """
        taking an action, either randomly or according to the Q-table
        """

        empty_cells = [k for k, v in enumerate(observation) if v == " "]
        if len(empty_cells) > 0:
            # take random action with probability epsilon
            # in the beginning when q-table is empty a random action is taken, so in the beginning the exploration probability is higher than epsilon
            state_key = observation
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
            action2qvalue = self.q_table[state_key]

            # Degrade epsilon values - change value of modulus wrt to number of games played
            if game % 10000 == 0:
                self.epsilon -= 0.002

            # Always plays first 20% of games as random - change value multiplied by num_games to set %
            if game <= int(0.2 * num_games):
                action = random.choice(empty_cells)
            elif random.random() < self.epsilon or len(action2qvalue) == 0:
                action = random.choice(empty_cells)
            else:  # probability of 1 - self.epsilon
                action, max_qv = max(action2qvalue.items(), key=get_value)
                # if max_qv>0:
                #     print("q-learning action:", action)
                #     print("max_qv:", max_qv)
        else:
            action = None

        return action

    def q_table_update(
        self,
        observation: str,  # S
        action: int,  # A
        new_observation: str,  # S'
        reward: float,  # R
    ) -> None:
        # update the Q-value for the action taken
        # Q(s, a) = (1 - alpha) * Q(s, a) + alpha * (reward + gamma * max_a'(Q(s', a')))

        state_key = observation
        new_state_key = new_observation
        if new_state_key in self.q_table and len(self.q_table[new_state_key]) > 0:
            next_action2qvalue = self.q_table[new_state_key]
            _, max_next_qv = max(next_action2qvalue.items(), key=get_value)
        else:
            max_next_qv = 0
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0}
        elif action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0
        self.q_table[state_key][action] = (1 - self.alpha) * self.q_table[state_key][action] + self.alpha * \
                                          (reward + self.gamma * max_next_qv)  # -self.q_table[state_key][action]
        self.q_table_update_count[state_key] = (
            self.q_table_update_count.get(state_key, 0) + 1
        )