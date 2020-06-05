import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_ind
from utils.file import get_file_paths


class DataPreprocess:
    def __init__(self, data_dir: str, top_n_gene: list, gene_limit):
        # Data folder path which includes train-test dataset and class file
        self.data_dir = data_dir
        self.top_n_gene = top_n_gene
        self.gene_limit = gene_limit
        # Getting file names to dict and accessing them just typing name of file eg. 'train', 'test'
        self.file_paths = get_file_paths(self.data_dir)
        # Read train-test datasets and class file
        self.train, self.test = self.read_data()
        self.classes, self.encoder = self.read_classes()

        self.t_test_result = self.data_preprocess()
        self.top_n_values, self.top_n_class = self.get_top_n()
        self.save_top_n()

    def data_preprocess(self):
        self.remove_fold_data(5)
        self.threshold_data()
        self.remove_low_variance()
        return self.calculate_t(save_df=True)

    def read_data(self):
        # Reading train and test csv files and converting them into numpy
        # For now only selecting first 100 genes. It helps working faster for further steps
        if self.gene_limit is None:
            train = pd.read_csv(self.file_paths['train']).to_numpy().T[:, :]
            test = pd.read_csv(self.file_paths['test']).to_numpy().T[:, :]
        else:
            train = pd.read_csv(self.file_paths['train']).to_numpy().T[:, :self.gene_limit]
            test = pd.read_csv(self.file_paths['test']).to_numpy().T[:, :self.gene_limit]
        return train, test

    def read_classes(self):
        # Open txt file as read mode
        file = open(self.file_paths['class'], "r")
        # Reading file contents as a list
        contents = file.read()
        # Separate new-lines
        classes = contents.split(sep="\n")[1:-1]
        # Creating encoder for classes
        le = LabelEncoder()
        # Fit all patient classes into label encoder and save them in new variable
        encoded = le.fit_transform(classes)
        return encoded, le

    def remove_fold_data(self, fold_n):
        # Loop over genes with all samples to find the index of genes that do not have enough fold
        genes_to_delete = [idx for idx, genes_row in enumerate(self.train.T)
                           if np.max(genes_row[1:]) < fold_n * np.min(genes_row[1:])]

        # Delete gene columns from training and test data
        self.train = np.delete(self.train, genes_to_delete, 1)
        self.test = np.delete(self.test, genes_to_delete, 1)

    def remove_low_variance(self):
        # Loop over genes with all samples to find the index of genes that do not have enough fold
        genes_to_delete = [idx for idx, genes_row in enumerate(self.train.T)
                           if np.std(genes_row[1:]) < 10]

        # Delete gene columns from training and test data
        self.train = np.delete(self.train, genes_to_delete, 1)
        self.test = np.delete(self.test, genes_to_delete, 1)

    def threshold_data(self):
        # Loop over genes with all samples to find the index of genes that do not have enough fold
        genes_to_delete = [idx for idx, genes_row in enumerate(self.train.T)
                           if np.max(genes_row[1:]) < 20 or np.min(genes_row[1:]) > 12000]

        # Delete gene columns from training and test data
        self.train = np.delete(self.train, genes_to_delete, 1)
        self.test = np.delete(self.test, genes_to_delete, 1)

    def calculate_t(self, save_df=False):
        # Placeholder for all class individual t test result
        total_t_result = []
        print(f'T-test Started on {len(set(self.classes))} class with {self.train.shape[1]} genes.\n')
        # Loop over all classes
        for cls in range(len(set(self.classes))):
            print(f'T-test on Class: {self.encoder.inverse_transform((cls,))[0]}')
            # Append class-based results in other list
            cls_t_result = []
            # Get indices of classes
            samp = np.where(self.classes == cls)[0] + 1
            # Take the first gene for t test
            for gene_0 in range(self.train.shape[1]):
                # Calculate t and p values when testing with all the remaining genes
                for gene_1 in range(gene_0 + 1, self.train.shape[1]):
                    t_value, p_value = ttest_ind(self.train[samp, gene_0], self.train[samp, gene_1])
                    cls_t_result.append((self.encoder.inverse_transform((cls,))[0],
                                         self.train[0, gene_0], gene_0, self.train[0, gene_1], gene_1,
                                         t_value, p_value))
            total_t_result.append(cls_t_result)

        # If desired, save these results in an additional file
        if save_df:
            path = os.path.join(self.data_dir, "pp5i_t_result.gr.csv")
            cols = ['Class', 'Gene 1', 'Indices of Gene 1', 'Gene 2', 'Indices of Gene 2', 't-value', 'p-value']
            data = np.squeeze(np.array(total_t_result).reshape((1, -1, 7)))
            df = pd.DataFrame(data, columns=cols)
            df.to_csv(path, index=False)
        print('\nT-test completed!')
        return np.array(total_t_result)

    def get_top_n(self):
        # Placeholder for top-n genes and their values
        top_n_values = {}
        top_n_genes = {}
        for n in self.top_n_gene:
            # Placeholder for top genes in max n gene
            n_train_list = []
            n_class_list = []
            # Loop over results and classes
            for encoded_class, cls_t_result in enumerate(self.t_test_result):
                indice_list = []
                cls_t_result = np.array(sorted(cls_t_result, key=lambda x: np.abs(float(x[5])), reverse=True))
                for ind_0, ind_1 in cls_t_result[:, [2, 4]]:
                    if ind_0 not in indice_list:
                        indice_list.append(int(ind_0))
                    if len(indice_list) == n:
                        break
                    if ind_1 not in indice_list:
                        indice_list.append(int(ind_1))
                    if len(indice_list) == n:
                        break

                samples = np.where(self.classes == encoded_class)[0] + 1
                n_train_list.append(self.train[np.min(samples):np.max(samples) + 1, indice_list])
                indice_list = list(self.train[0, indice_list])
                indice_list.append(self.encoder.inverse_transform((encoded_class,))[0])
                n_class_list.append(indice_list)
            n_class_list = np.array(n_class_list)
            n_train_list = np.concatenate(n_train_list, axis=0)
            n_train_list = np.concatenate((n_train_list, self.encoder.inverse_transform(np.sort(self.classes)).reshape((-1, 1))),
                                          axis=1)

            top_n_values[n] = (n_train_list)
            top_n_genes[n] = (n_class_list)

        return top_n_values, top_n_genes

    def save_top_n(self):
        save_dir = os.path.join(self.data_dir, "top_n")
        if not os.path.exists(save_dir):
            os.rmdir(save_dir)
            os.mkdir(save_dir)
        for n, (n_values, n_genes) in enumerate(zip(self.top_n_values.values(), self.top_n_class.values())):
            cols = [f'Gene: {cnt}' for cnt in range(self.top_n_gene[n])]
            cols.append('Class')
            df_values = pd.DataFrame(n_values, columns=cols)
            path = os.path.join(save_dir, f"pp5i_train.top{self.top_n_gene[n]}.values.gr.csv")
            df_values.to_csv(path, index=False)
            df_genes = pd.DataFrame(n_genes, columns=cols)
            path = os.path.join(save_dir, f"pp5i_train.top{self.top_n_gene[n]}.genes.gr.csv")
            df_genes.to_csv(path, index=False)
        print(f'Files saved to: {save_dir}')


if __name__ == "__main__":
    dp = DataPreprocess('data', [2, 4, 6, 8, 10, 15, 20, 25])
