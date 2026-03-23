import pandas as pd
import os


def load_data(filename):
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, f'{filename}'), index_col=0)

    return data
