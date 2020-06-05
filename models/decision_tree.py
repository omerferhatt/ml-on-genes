import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


def train_decision_tree(x_train, y_train, fold):
    dt = DecisionTreeClassifier()
    scores = cross_val_score(dt, x_train, y_train, cv=fold)
    print(f"Decision Tree\n"
          f"\tAccuracy: %0.3f\n" % scores.mean())

    return scores.mean()
