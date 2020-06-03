# Main libraries to manipulate arrays and reading data files
import os
from datetime import datetime
import numpy as np
import pandas as pd
# Library imports for file handling
from tqdm import trange
# Importing machine learning framework
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_ind
# Importing custom user libraries
from utils.file import get_fp_dict, read_cls_txt


class DataPreprocess:
    def __init__(self, data_path, top_gen=(2, 4, 6, 8, 10, 12, 15, 20, 25, 30)):
        self.data_dir = data_path
        self.top_gen_pair = top_gen
        self.file_paths = get_fp_dict(self.data_dir)
        self.classes = read_cls_txt(self.file_paths['class'])
        self.data = self.read_data()

        self.top_gen_index, self.top_gen_index_df = self.get_top_gen_index()

    def read_data(self):
        train = pd.read_csv(self.file_paths['train']).to_numpy().T[1:, :100]
        test = pd.read_csv(self.file_paths['test']).to_numpy().T[1:, :100]
        return {"train": train, "test": test}

    def get_top_gen_index(self):
        encoder, encoded_class = self._encode_label(self.classes)
        threshold_data = self._threshold_genes(self.data)
        ttest_res = self._calculate_t(threshold_data['train'], encoded_class)
        top_gen_pair = self._get_top_gen_pairs(ttest_res, self.top_gen_pair)
        top_gen_index = self._encode_pairs(ttest_res, top_gen_pair)
        top_gen_index_df = self._create_top_gen_df(top_gen_index, encoder, self.classes)
        return top_gen_index, top_gen_index_df

    def save_top_gen(self):
        now = datetime.now()
        date_time = now.strftime("%m_%d_%y_%H_%M_%S")
        file_path = os.path.join(self.data_dir, f"pp5i_train.topN.gr.{date_time}.csv")
        self.top_gen_index_df.to_csv(file_path)

    @staticmethod
    def _encode_label(arr):
        le = LabelEncoder()
        arr = le.fit_transform(arr)
        return le, arr

    @staticmethod
    def _threshold_genes(arr):
        for d in list(arr.values()):
            for row in range(d.shape[0]):
                for col in range(d.shape[1]):
                    if d[row, col] < 20 or d[row, col] > 16000:
                        d[row, col] = 0
        return arr

    @staticmethod
    def _calculate_t(arr, classes):
        t_res = []
        for cls in trange(np.max(classes)+1, desc="Classes"):
            sub_class = []
            samp = np.where(classes == cls)[0]
            for gene in trange(arr.shape[1], desc=f"Genes in class {cls}"):
                for other_gene in range(gene + 1, arr.shape[1]):
                    t_val, p_val = ttest_ind(arr[samp, gene], arr[samp, other_gene])
                    sub_class.append((gene, other_gene, (t_val, p_val)))
            t_res.append(sub_class)
        return np.array(t_res)

    @staticmethod
    def _get_top_gen_pairs(t_res, top_list):
        main_list = {}
        for t in top_list:
            tot_cls = []
            for cls in t_res:
                cls_list = []
                for ind, (t_val, p_val) in enumerate(cls[:, 2][:]):
                    if p_val < 0.005:
                        cls_list.append([np.abs(t_val), ind])
                cls_list = np.array(cls_list)
                cls_list = cls_list[cls_list[:, 0].argsort()]
                tot_cls.append(cls_list[-t:])
            main_list[t] = tot_cls
        return main_list

    @staticmethod
    def _encode_pairs(ttest_res, top_dict):
        top_gen_list = []
        for top, keys in zip(list(top_dict.values()), list(top_dict.keys())):
            gen_ind = []
            for cls_top, cls_res in zip(top, ttest_res):
                temp_list = []
                for pairs in cls_top[:, 1]:
                    temp_list.append(np.array(cls_res[int(pairs), [0, 1]]))
                gen_ind.append(list(set(list(np.array(temp_list).reshape(1, -1)[0]))))
            top_gen_list.append(gen_ind)
        return np.array(top_gen_list)

    def _create_top_gen_df(self, top_gen, enc, classes):
        rows = enc.inverse_transform(np.arange(top_gen.shape[1]))
        cols = [f"Top gen from {pair} pair" for pair in self.top_gen_pair]
        df = pd.DataFrame(data=top_gen.T, index=rows, columns=cols)
        return df
