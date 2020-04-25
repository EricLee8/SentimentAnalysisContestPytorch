import os
import argparse

parser = argparse.ArgumentParser(description='Train all shell')
parser.add_argument('--large', default=False, type=bool, help='whether to use roberta-large')
args = parser.parse_args()

print("========================== Training kfold_cnn_avg.py ===============================")
os.system("python3 kfold_cnn_avg.py" + (" --zhlarge True" if args.large else ""))
print("========================== Training Done ===============================\n")

print("\n========================== Training kfold_gru.py ===============================")
os.system("python3 kfold_gru.py" + (" --zhlarge True" if args.large else ""))
print("========================== Training Done ===============================\n")

print("\n========================== Training kfold_kmax.py ===============================")
os.system("python3 kfold_kmax_avg.py" + (" --zhlarge True" if args.large else ""))
print("========================== Training Done ===============================\n\n")

