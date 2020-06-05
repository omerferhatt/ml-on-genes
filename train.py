import numpy as np
import pandas as pd

from models.decision_tree import train_decision_tree
from models.dnn import train_dnn
from models.knn import train_knn
from models.naive_bayes import train_gaussian_nb
from models.random_forest import train_random_forest, train_improved_random_forest
from utils.file import get_top_n


class Training:
    def __init__(self, top_n_list, top_n_path='data', random_seed=3):
        self.top_n_list = top_n_list
        self.top_n_path = top_n_path
        self.random_seed = random_seed

        self.dataset = self.read_dataset()
        self.best_classifier_list = self.training()
        self.best_classifier, self.best_accuracy, self.best_n = self.get_best_classifier()
        self.print_result()
        self.improved_model()

    def read_dataset(self):
        return get_top_n(self.top_n_path, self.top_n_list)

    def shuffle(self, array):
        np.random.seed(self.random_seed)
        np.random.shuffle(array)

    def training(self):
        best_classifier_all = []
        for n, dataset_path in self.dataset:
            dataframe = pd.read_csv(dataset_path, index_col=False)
            dataset = dataframe.to_numpy()
            self.shuffle(dataset)

            fold = 6
            x_train = dataset[:, :-1]
            y_train = dataset[:, -1]
            print(f"Top: {n} Genes")
            print(f"Cross-validation fold: {fold}")
            print("-------------------------------")
            score_nb = train_gaussian_nb(x_train, y_train, fold)
            score_dt = train_decision_tree(x_train, y_train, fold)
            score_dnn = train_dnn(x_train, y_train, fold)
            score_rf = train_random_forest(x_train, y_train, fold)
            score_knn_2, score_knn_3, score_knn_4 = train_knn(x_train, y_train, [2, 3, 4], fold)
            score_list = np.array([score_nb, score_dt, score_dnn, score_rf, score_knn_2, score_knn_3, score_knn_4])
            max_acc = np.max(score_list)
            best_classifier_idx = np.where(score_list == max_acc)
            best_classifier = self.find_best_classifier(best_classifier_idx[0])
            best_classifier_all.append([best_classifier, max_acc, n])
            print(f"Maximum accuracy: {max_acc:0.4f}")
            print("\n\n")
            print("-------------------------------")
        return best_classifier_all

    @staticmethod
    def find_best_classifier(index):
        if index == 0:
            print("Best classifier: Naive Bayes")
            return "Naive Bayes"
        elif index == 1:
            print("Best classifier: Decision tree")
            return "Decision Tree"
        elif index == 2:
            print("Best classifier: Neural Network")
            return "Neural Network"
        elif index == 3:
            print("Best classifier: Random Forest")
            return "Random Forest"
        elif index == 4:
            print("Best classifier: KNN with 2 neighbor")
            return "KNN with 2 neighbor"
        elif index == 5:
            print("Best classifier: KNN with 3 neighbor")
            return "KNN with 3 neighbor"
        elif index == 6:
            print("Best classifier: KNN with 4 neighbor")
            return "KNN with 4 neighbor"

    def get_best_classifier(self):
        sorted_list = sorted(self.best_classifier_list, key=lambda x: x[1], reverse=True)
        return sorted_list[0][:3]

    def print_result(self):
        print(f"\n\n************************\n"
              f"Best classifier of the all training is:\n"
              f"\t{self.best_classifier}\n"
              f"Accuracy of the classifier is:\n"
              f"\t{self.best_accuracy}\n"
              f"Best top gene set is:\n"
              f"\t{self.best_n}\n"
              f"************************\n")

    def improved_model(self):
        dataframe = pd.read_csv(self.dataset[-1][1], index_col=False)
        dataset = dataframe.to_numpy()
        self.shuffle(dataset)

        fold = 6
        x_train = dataset[:, :-1]
        y_train = dataset[:, -1]

        score_rf_imp = train_improved_random_forest(x_train, y_train, fold)

        print(f"\n\n************************\n"
              f"Accuracy of the improved classifier is:\n"
              f"\t{score_rf_imp}\n"
              f"************************\n")


if __name__ == '__main__':
    tr = Training([2, 4, 6, 8, 10, 12, 15, 20, 25, 30])
