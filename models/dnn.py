import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score


def train_dnn(x_train, y_train, fold):
    dnn = MLPClassifier(max_iter=600)
    scores = cross_val_score(dnn, x_train, y_train, cv=fold)
    print(f"Neural Network\n"
          f"\tAccuracy: %0.3f\n" % scores.mean())

    return scores.mean()