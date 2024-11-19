import argparse
import os
import numpy as np
import pandas as pd
from polyleven import levenshtein


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str, help="Path to ground-truth data.")
    parser.add_argument("--output_file", type=str, help="Output filepath.")
    parser.add_argument("--min_dist", type=int, help="Min mutation distance to 99th percentile.")
    parser.add_argument(
        "--filter_percentile", nargs=2, type=float, help="Fitness percentile to take."
    )
    parser.add_argument("--top_quantile", type=float, default=0.99, help="Top quantile.")
    args = parser.parse_args()
    return args


def filter_seq(data_df, percentile, min_mutant_dist, top_quantile=0.99) -> pd.DataFrame:

    lower_value = data_df.target.quantile(percentile[0])
    upper_value = data_df.target.quantile(percentile[1])
    top_quantile = data_df.target.quantile(top_quantile)
    top_sequences_df = data_df[data_df.target >= top_quantile]
    filtered_df = data_df[data_df.target.between(lower_value, upper_value)]

    get_min_dist = lambda x: np.min(
        [levenshtein(x.strip(), top_seq.strip()) for top_seq in top_sequences_df.sequence]
    )
    mutant_dist = filtered_df.sequence.apply(get_min_dist)

    return filtered_df[mutant_dist >= min_mutant_dist]


def main(args):
    assert os.path.exists(args.csv_file), f"{args.csv_file} does not exist."
    df = pd.read_csv(args.csv_file)
    filtered_df = filter_seq(df, args.filter_percentile, args.min_dist, args.top_quantile)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    filtered_df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
