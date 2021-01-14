import pandas as pd
import argparse
from pathlib import Path


def calculate_baseline(path, value):
    file_path = Path(path)
    csv = pd.read_csv(file_path, usecols=(value, ))
    mean = csv.mean()
    values_without_mean = (csv - mean).abs()
    baseline = values_without_mean.sum() / len(values_without_mean)
    print("Mean value of", value, "is :", mean.iloc[0])
    print("The baseline of", value, "is:", baseline.iloc[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True)
    parser.add_argument('-v', '--value', required=True)
    args = parser.parse_args()
    calculate_baseline(args.file, args.value)