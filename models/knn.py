import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def train_knn(x_train, y_train, neighbors, fold):
    scores = []
    print(f"KNN")
    for neighbor in neighbors:
        knn = KNeighborsClassifier(n_neighbors=neighbor)
        score = cross_val_score(knn, x_train, y_train, cv=fold)
        print(f"\tn:{neighbor} Accuracy: {score.mean():0.3f}")
        scores.append(score.mean())
    print("\n")
    return scores
