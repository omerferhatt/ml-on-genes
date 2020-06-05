import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score


def train_gaussian_nb(x_train, y_train, fold):
    nb = GaussianNB()
    scores = cross_val_score(nb, x_train, y_train, cv=fold)
    print(f"Naive Bayes\n"
          f"\tAccuracy: %0.3f\n" % scores.mean())

    return scores.mean()
