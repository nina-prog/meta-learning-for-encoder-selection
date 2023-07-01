import pandas as pd
import os


def add_dataset_meta_information(df=None, path_to_meta_df=None, nan_threshold=0.4, replacing_strategy="mean"):
    """
    Adds the meta information to the dataset. 
    The meta dataset contains null values, since not all datasets contain all meta information. 
    Therefore, one can specify a threshold. If a column has more than threshold % of nan values it will be dropped. 
    The remaining nan values can be filled with either the 'mean', 'median' or a given constant.

            Parameters:
                    df (pandas.DataFrame): The basic dataframe to which the meta information will be added. 
                    
                    path_to_meta_df (string): The path from which the meta dataframe will be loaded
                    
                    nan_threshold (float): Has to be in the range of [0, 1]. Columns, which have more nan values than nan_threshold % will be dropped. 
                    
                    replacing_strategy ("mean", "median" or float): The strategy how to replace the reamining nan values. 

            Returns:
                    df (pandas.DataFrame): The dataframe df, which is extended by the meta information. The datasets are merged based on the dataset_id. 
    """
    # Validate option
    assert isinstance(replacing_strategy, float) or isinstance(replacing_strategy, int) or replacing_strategy in ['mean', 'median']
    assert 0 <= nan_threshold <= 1
    assert os.path.exists(path_to_meta_df)
    assert "dataset" in df.columns
    
    # Load the meta dataset
    dataset_agg = pd.read_csv(path_to_meta_df, index_col=0)
    
    
    # Get column names where the percentage of nan values is greater than the threshold
    total_number_of_rows = dataset_agg.shape[0]
    sum_of_nan_values = dataset_agg.isna().sum()
    ratio_of_nan_values = sum_of_nan_values / total_number_of_rows
    selector_of_columns = ratio_of_nan_values[(ratio_of_nan_values > nan_threshold)]
    column_names_to_drop = list(selector_of_columns.index)

    # Drop the columns
    dataset_agg_filtered = dataset_agg.drop(column_names_to_drop, axis=1)
    
    # Fill the remaining nan values with the mean or median
    if replacing_strategy == "mean":
        dataset_agg_filtered = dataset_agg_filtered.fillna(dataset_agg_filtered.mean())
    elif replacing_strategy == "median":
        dataset_agg_filtered = dataset_agg_filtered.fillna(dataset_agg_filtered.median())
    else:
        # Only float or integer is possible
        dataset_agg_filtered = dataset_agg_filtered.fillna(replacing_strategy)
    
    # Merge datasets
    dataset_agg_filtered = dataset_agg_filtered.rename(columns={"dataset_id": "dataset"})
    merged_df = df.merge(dataset_agg_filtered, how="left", on=["dataset"])

    # Drop the dataset column
    # merged_df = merged_df.drop(["dataset"], axis=1)

    return merged_df
