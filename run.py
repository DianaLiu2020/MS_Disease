import argparse
import pandas as pd
from analysis import MS


def run(args_dict):
    # Load dataframes
    df = pd.read_csv(args_dict['input'])
    test = pd.read_csv(args_dict['test'])

    # Instantiate predictive analysis
    ms = MS(df, test)
    ms.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predictive Analytics for MS Disease.')
    parser.add_argument('-i', '--input', required=True, help='Path to input file')
    parser.add_argument('-t', '--test', required=True, help='Path to test file with new obs')
    args_dict = vars(parser.parse_args())
    run(args_dict)
