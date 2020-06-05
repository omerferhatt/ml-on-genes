import numpy as np
import pandas as pd

from utils.file import get_file_paths


def read_data(data_dir):
    file_paths = get_file_paths(data_dir)

    train = pd.read_csv(file_paths['train']).to_numpy().T[1:, :200]
    test = pd.read_csv(file_paths['test']).to_numpy().T[1:, :200]
    return train, test
