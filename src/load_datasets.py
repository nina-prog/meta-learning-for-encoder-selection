import pandas as pd

from pathlib import Path
from typing import Union


def load_dataset(path: Union[Path, str], verbosity=1, subsample=None) -> pd.DataFrame:
    if verbosity > 0: print("Loading data ...")
    # Read dataframe
    df = pd.read_csv(path, index_col=0)

    # Take subsample, if provided in the args
    if subsample is not None:
        print(f"Taking first {subsample} rows ...")
        df = df.iloc[:subsample]

    return df


def load_rankings(path: Union[Path, str], verbosity=1, subsample=None) -> pd.DataFrame:
    if verbosity > 0: print("Loading rankings ...")
    # Read dataframe
    out = pd.read_csv(path, index_col=0, header=[0, 1, 2, 3])
    out.columns.name = ("dataset", "model", "tuning", "scoring")

    # Take subsample, if provided in the args
    if subsample is not None:
        print(f"Taking first {subsample} rows ...")
        out = out.iloc[:subsample]


    return out

