import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def get_plot(game_data_dict):
    num_games = len(game_data_dict)
    # to determine plot size created col/row size
    num_cols = int(np.ceil(np.sqrt(num_games)))  # calculate number of columns based on square root of number of games
    num_rows = (num_games + num_cols - 1) // num_cols  # calculate number of rows based on number of columns

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))

    for i, (game_numer, game) in enumerate(game_data_dict.items()):
        row_index = i // num_cols
        col_index = i % num_cols

        ax = axes[row_index, col_index]
        try:
            sns.heatmap(([game[v] for v in range(0, 3)], [game[v] for v in range(3, 6)], [game[v] for v in range(6, 9)]),
                        annot=True, cmap='coolwarm', square=True, linewidths=.5, cbar=True, fmt=".2f", ax=ax)
        except ValueError:
            print(f"Skipping game {game_numer} due to non-numeric data")
            continue
        ax.set_xticks(np.arange(3) + 0.5)
        ax.set_yticks(np.arange(3) + 0.5)
        ax.set_xticklabels(['1', '2', '3'])
        ax.set_yticklabels(['1', '2', '3'], rotation=0)
        ax.set_title(f"Game {game_numer}")

    fig.suptitle("Q-Table Updated Values", fontsize=16)
    plt.tight_layout(pad=2)  # add space between subplots
    plt.show()

