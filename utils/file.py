# Library imports for file handling
import glob
import os
import pandas as pd

def get_file_paths(path):
    # Creates file path dict. Searches for "train", "test" and "class" substrings.
    files = glob.glob(os.path.join(path, '*.*'))
    # Creating empty file dictionary to hold all required file paths in it.
    file_paths = {}
    for file in files:
        if 'train' in file and 'csv' in file:
            file_paths['train'] = file
        elif 'test' in file and 'csv' in file:
            file_paths['test'] = file
        elif 'class' in file and 'txt' in file:
            file_paths['class'] = file
    return file_paths


def get_top_n(path, top_n_list):
    # Creates file path dict. Searches for "topN" substrings.
    path_files = os.path.join(path, 'top_n/*.*')
    files = glob.glob(path_files)
    # Creating empty file dictionary to hold all required file paths in it.
    values_paths = {}
    for file in files:
        for n in top_n_list:
            if f'top{int(n):02}' in file and 'csv' in file:
                values_paths[n] = file
    return sorted(values_paths.items(), key=lambda x: int(x[0]))

def create_best_test():
    best_train_path = "data/pp5i_train.best30.csv"
    best_test_path = "data/pp5i_test.best30.csv"
    test_path = "data/pp5i_test.gr.csv"

    best_train_data = pd.read_csv(best_train_path, index_col=False)
    best_train_data_cols = list(best_train_data.columns)[:-1]
    test_data = pd.read_csv(test_path, index_col=False).transpose().to_numpy()
    test_data = pd.DataFrame(test_data[1:, :], columns=test_data[0, :])
    best_test_data = test_data[best_train_data_cols]

    best_test_data.to_csv(best_test_path, index=False)