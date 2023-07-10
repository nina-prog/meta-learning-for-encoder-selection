import pandas as pd

from functools import reduce
from itertools import product

import src.evaluate_regression as er


def get_pairwise_target(df, features, target, column_to_compare):
    dfs = []
    for e1, e2 in product(df[column_to_compare].unique(), repeat=2):
        if e1 >= e2:
            continue
        tmp = pd.merge(df.loc[df[column_to_compare] == e1], df.loc[df[column_to_compare] == e2],
                       on=features, how="inner", suffixes=("_1", "_2"))
        tmp[(e1, e2)] = (tmp[f"{target}_1"] < tmp[f"{target}_2"]).astype(int)
        tmp[(e2, e1)] = (tmp[f"{target}_2"] < tmp[f"{target}_1"]).astype(int)
        dfs.append(tmp[features + [(e1, e2), (e2, e1)]])
    return reduce(lambda d1, d2: pd.merge(d1, d2, on=features, how="outer"), dfs)


def join_pairwise2rankings(X: pd.DataFrame, y: pd.DataFrame, factors):
    """
    Fixed ["dataset", "model", "tuning", "scoring"], the rank of an encoder is the _opposite_ of the number
    of encoders that it beats; the more other encodes it beats, the better it is.
    """

    if not y.index.equals(X.index):
        raise ValueError("y and X must have the same index.")

    tmp = pd.concat([X, y], axis=1).melt(id_vars=factors, var_name="encoder_pair",
                                         value_name="is_better_than")
    tmp = pd.concat([tmp.drop("encoder_pair", axis=1),
                     pd.DataFrame(tmp["encoder_pair"].to_list(), columns=["encoder", "encoder_2"])],
                    axis=1)
    tmp = tmp.groupby(factors + ["encoder"])["is_better_than"].sum().reset_index()
    tmp["rank_pred"] = er.score2ranking(tmp["is_better_than"], ascending=False)

    return tmp.drop("is_better_than", axis=1)
