# Library imports for file handling
import glob
import os


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
