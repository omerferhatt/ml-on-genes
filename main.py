import argparse

from utils.data_preprocess import DataPreprocess
from train import Training
from predict import evaluate

parser = argparse.ArgumentParser(description="Disease class prediction with using genetic micro-array data")
parser.add_argument("-p", "--pre-data",
                    action="store_true", dest="data_proc",
                    help="When the flag is activated, it performs data pre-processing.")
parser.add_argument("-l", "--limit-gene", nargs=1,
                    action="store", default=0, dest="limit",
                    help="Limit genes in all datasets, it speeds up on data pre-processing development.")
parser.add_argument("-t", "--train",
                    action="store_true", dest="train",
                    help="When the flag is activated, it performs training.")
parser.add_argument("-e", "--evaluate",
                    action="store_true", dest="evaluate",
                    help="When the flag is activated, it predicts test classes.")

args = parser.parse_args()

if args.data_proc:
    dp = DataPreprocess("data", [2, 4, 6, 8, 10, 12, 15, 20, 25, 30], int(args.limit[0]))

if args.train:
    tr = Training([2, 4, 6, 8, 10, 12, 15, 20, 25, 30])

if args.evaluate:
    evaluate()
