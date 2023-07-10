

def get_pearson_correlated_features(data=None, threshold=0.7):
    """
    Calculates the pearson correlation of all features in the dataframe and returns a set of features with a
    correlation greater than the threshold.

    :param data: The input dataframe.
    :type data: pd.DataFrame
    :param threshold: The threshold for the correlation coefficient in the range of [0.0, 1.0].
    :type threshold: float,optional(default=0.7)

    :return: The set of features with a correlation greater than the threshold.
    :rtype: set
    """
    # Calculate correlation matrix
    corr_matrix = data.corr()

    # Get the set of correlated features
    correlated_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                correlated_features.add(colname)

    return correlated_features


def drop_pearson_correlated_features(train_data=None, test_data=None, threshold=0.7, verbosity=0):
    if verbosity > 0:
        print(f"Drop pearson correlated features with threshold {threshold}...")
    correlated_features = get_pearson_correlated_features(data=train_data, threshold=threshold)
    
    # Filter out some features
    if verbosity > 0:
        print("Filter correlated features")
    correlated_features = [f for f in correlated_features if not f.startswith("enc_dim_")]
    correlated_features = [f for f in correlated_features if not f.startswith("model_")]
    correlated_features = [f for f in correlated_features if not f.startswith("tuning_")]
    correlated_features = [f for f in correlated_features if not f.startswith("scoring_")]

    # Drop features
    train_data = train_data.drop(correlated_features, axis=1)
    if test_data is not None:
        test_data = test_data.drop(correlated_features, axis=1)
    
    return train_data, test_data
