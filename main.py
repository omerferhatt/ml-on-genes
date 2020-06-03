import argparse

from utils.data_preprocess import DataPreprocess


parser = argparse.ArgumentParser(description="Disease class prediction with using genetic micro-array data")
parser.add_argument("-p", "--pre-data",
                    action="store_true", dest="data_proc",
                    help="When the flag is activated, it performs data pre-processing.")

args = parser.parse_args()

if args.data_proc:
    dp = DataPreprocess("data")
    dp.save_top_gen()

