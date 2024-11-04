from __future__ import annotations

import json
import random
import numpy as np
import gymnasium as gym
from gymnasium.core import ObsType
from typing import Any
from tqdm import tqdm
from plot import get_plot
from copy import deepcopy
from agent import QLearningAgent

# Utils
from utils import take_foto, game_state, assign_reward, GameStatus, human_agent


class TictactoeEnvironment(gym.Env):
    def __init__(self):
        self.board = {a: " " for a in range(9)}
        self.reward = 0

    def step(self, action: int) -> tuple[str, float, bool, bool, dict[str, Any]]:
        self.board[action] = "X"
        # self.board = agent.agent_action(board)
        # Randomly put O into empty cell
        empty_cells = [k for k, v in self.board.items() if v == " "]
        if len(empty_cells) > 0:
            self.board[random.choice(empty_cells)] = "O"
        state = game_state(take_foto(self.board))
        done = state != GameStatus.ONGOING
        self.reward = assign_reward(state)
        observation = take_foto(self.board)
        return observation, self.reward, done, False, {}

    def render(self):
        visualitation = np.array(list(self.board.values())).reshape(3, 3)
        print("---------------")
        print(visualitation)
        print("---------------")
        print("Reward: ", self.reward)

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        board = {a: " " for a in range(9)}
        self.board = board
        observation = take_foto(self.board)
        return observation, {}


def main():
    # state -> action -> new-state
    env = TictactoeEnvironment()
    q_agent = QLearningAgent(9)
    num_games = 2000000
    bucket_size = 20000
    winner_stats = [0 for _ in range(int(num_games / bucket_size))]
    # print(env.render())

    game_data_dict = {}
    for game in tqdm(range(num_games)):  # tqdm for progres bar
        board, _ = env.reset()
        for k in range(9):
            # Get the agent's action randomly
            # action = agent.agent_action(env.board)

            # Get the agent's action using Q-learning
            action = q_agent.agent_action(board, game, num_games)
            assert action is not None

            # # Human Interactive Player - comment out this section to avoid human player
            # if game % 25000 == 0 and game != 0:  # test case to play human player for now
            #     env.render()
            #     action = human_agent(board)
            #     while action is "Invalid":
            #         env.render()
            #         action = human_agent(board)

            # Apply the action to the environment
            new_board, reward, done, _, _ = env.step(action)
            # print(f"old: {board}")
            # print(f"action: {action}")
            # print(f"new: {new_board}")

            q_agent.q_table_update(board, action, new_board, reward)

            board = new_board
            if done:
                # print(game_state(new_board))
                winner_stats[int(np.floor(game / bucket_size))] += int(
                    game_state(new_board) == GameStatus.X_WON
                )
                break

        # To plot values after each game
        for board, v in q_agent.q_table.items():
            if game % 400000 == 0 and len(v) == 9:  # take only initial board values and set how many plots needed(% value)
                print(v)
                game_data_dict[game] = deepcopy(v)

    # plot board probability
    get_plot(game_data_dict)

    print(env.render())
    print(winner_stats)

    with open("qtable.jsonl", 'w') as f:
        for board, v in q_agent.q_table.items():
            nice_board = (
                    board[0] + "|"
                    + board[1] + "|"
                    + board[2]
                    + "\n" + "-----" + "\n"  # adding line-breaks so it looks nicer!
                    + board[3] + "|"
                    + board[4] + "|"
                    + board[5]
                    + "\n" + "-----" + "\n"
                    + board[6] + "|"
                    + board[7] + "|"
                    + board[8]
            )
            f.write(f"--- board:\n{nice_board}\n--- action2values:\n{v}\n ---update count: {q_agent.q_table_update_count[board]}\n")


if __name__ == "__main__":
    main()
