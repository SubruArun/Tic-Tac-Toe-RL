from enum import Enum
from typing import Union


class GameStatus(Enum):
    ONGOING = 0
    X_WON = 1
    O_WON = 2
    DRAW = 3

    @staticmethod
    def from_str(winner: str):
        return GameStatus.X_WON if winner == "X" else GameStatus.O_WON


def take_foto(board: dict[int, str]) -> str:
    observation = "".join(board.values())
    return observation


def game_state(board: str) -> GameStatus:
    for i in range(3):
        if board[i] == board[i + 3] == board[i + 6] != " ":
            return GameStatus.from_str(board[i])
        if board[i * 3] == board[i * 3 + 1] == board[i * 3 + 2] != " ":
            return GameStatus.from_str(board[i * 3])
    if board[0] == board[4] == board[8] != " ":
        return GameStatus.from_str(board[0])

    if board[2] == board[4] == board[6] != " ":
        return GameStatus.from_str(board[2])

    for i in range(9):
        if board[i] == " ":
            return GameStatus.ONGOING

    return GameStatus.DRAW


def assign_reward(state: GameStatus) -> float:
    if state is GameStatus.X_WON:
        return 1
    elif state is GameStatus.O_WON:
        return -1
    else:                                      # DRAW or ONGOING
        return 0


def get_value(item: tuple[int, float]) -> float:
    a, qv = item
    return qv


def human_agent(board: dict) -> Union[int, str]:
    valid = False
    while not valid:
        action = input("Enter your move (0-8): ")
        try:
            action = int(action)
            if action in range(9) and board[action] == " ":
                valid = True
            else:
                print("Invalid move. Try again.")
                return "Invalid"
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 8.")
            return "Invalid"
        return action
