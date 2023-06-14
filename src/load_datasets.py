import pandas as pd

from pathlib import Path
from typing import Union


def load_dataset(path: Union[Path, str], verbosity=1) -> pd.DataFrame:
    if verbosity > 0: print("Loading data ...")
    return pd.read_csv(path, index_col=0)


def load_rankings(path: Union[Path, str], verbosity=1) -> pd.DataFrame:
    if verbosity > 0: print("Loading rankings ...")
    out = pd.read_csv(path, index_col=0, header=[0, 1, 2, 3])
    out.columns.name = ("dataset", "model", "tuning", "scoring")
    return out

