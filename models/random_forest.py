import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def train_random_forest(x_train, y_train, fold):
    rf = RandomForestClassifier(random_state=10)
    scores = cross_val_score(rf, x_train, y_train, cv=fold)
    print(f"Random Forest\n"
          f"\tAccuracy: %0.3f\n" % scores.mean())

    return scores.mean()


def train_improved_random_forest(x_train, y_train, fold):
    rf = RandomForestClassifier(n_estimators=700, random_state=10,
                                min_samples_split=2, n_jobs=-1,
                                max_depth=140, max_features=12)
    scores = cross_val_score(rf, x_train, y_train, cv=fold)

    return scores.mean()


def evaluate_improved_random_forest(x_train, y_train, x_test):
    rf = RandomForestClassifier(n_estimators=700, random_state=10,
                                min_samples_split=2, n_jobs=-1,
                                max_depth=140, max_features=12)

    rf.fit(x_train, y_train)
    return rf.predict(x_test)