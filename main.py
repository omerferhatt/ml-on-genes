import argparse

from utils.data_preprocess import DataPreprocess


parser = argparse.ArgumentParser(description="Disease class prediction with using genetic micro-array data")
parser.add_argument("-p", "--pre-data",
                    action="store_true", dest="data_proc",
                    help="When the flag is activated, it performs data pre-processing.")
parser.add_argument("-l", "--limit-gene", nargs=1,
                    action="store", default=None, dest="limit",
                    help="Limit genes in all datasets, it speeds up on data pre-processing development.")

args = parser.parse_args(["--pre-data", "--limit-gene", "100"])

if args.data_proc:
    dp = DataPreprocess("data", [2, 4, 6, 8, 10, 12, 15, 20, 25, 30], int(args.limit[0]))

