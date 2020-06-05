import numpy as np
import pandas as pd

from utils.file import create_best_test
from models.random_forest import evaluate_improved_random_forest

def evaluate():
    create_best_test()
    train = pd.read_csv("data/pp5i_train.best30.csv", index_col=False).to_numpy()
    test = pd.read_csv("data/pp5i_test.best30.csv", index_col=False).to_numpy()
    np.random.seed(10)
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]

    pred = evaluate_improved_random_forest(x_train, y_train, test)
    print("Prediction result of test data")
    for i, res in enumerate(pred):
        print(f"Sample: {i} - {res}")
