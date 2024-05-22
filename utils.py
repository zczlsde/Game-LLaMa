import numpt as np
import pandas as pd
import markdown
import nashpy as nash

def extract_payoff_tables(matrix):
    """
    Extract the individual payoff tables for two players from the given matrix.
    """
    player1_payoffs = np.array([[cell[0] for cell in row] for row in matrix])
    player2_payoffs = np.array([[cell[1] for cell in row] for row in matrix])
    return player1_payoffs, player2_payoffs

def create_sorted_table(input_table):

    html_text = markdown.markdown(input_table, extensions=['tables'])
    tables = pd.read_html(html_text)
    out_table = tables[0]
    df = out_table.reindex(sorted(out_table.columns), axis=1)
    df.set_index(df.columns[0], inplace=True)
    df_sorted = df.sort_index()

    final_out_table = df_sorted.applymap(lambda x: tuple(map(int, x.strip("()").split(","))))
    return final_out_table.to_numpy()

def find_nash_equilibria(payoff_table_1, payoff_table_2):
    """
    Find Nash equilibria for two games.
    """
    game_1 = nash.Game(payoff_table_1[0], payoff_table_1[1])
    game_2 = nash.Game(payoff_table_2[0], payoff_table_2[1])
    
    equilibria_1 = list(game_1.support_enumeration())
    equilibria_2 = list(game_2.support_enumeration())
    
    return equilibria_1, equilibria_2

def convert_equilibria_to_hashable(equilibria):
    """
    Convert the Nash equilibria (which are numpy arrays) to hashable tuples.
    """
    return [tuple(map(tuple, eq)) for eq in equilibria]

def compare_nash_equilibria(matrix_1, matrix_2):
    """
    Compare Nash equilibria of two payoff matrices.
    """
    # Extract individual payoff tables for each player
    payoff_table_1 = extract_payoff_tables(matrix_1)
    payoff_table_2 = extract_payoff_tables(matrix_2)

    # Find Nash equilibria
    equilibria_1, equilibria_2 = find_nash_equilibria(payoff_table_1, payoff_table_2)
    # print(equilibria_1)
    # print(equilibria_2)
    total_equilibria = len(equilibria_1) * len(equilibria_2)

    # Find intersection of equilibria sets
    common_equilibria = set(convert_equilibria_to_hashable(equilibria_1)) & set(convert_equilibria_to_hashable(equilibria_2))

    return common_equilibria, len(common_equilibria), total_equilibria