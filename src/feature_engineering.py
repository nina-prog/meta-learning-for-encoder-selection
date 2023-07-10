import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression, f_regression


def normalize_train_data(X_train: pd.DataFrame, method: str = "standard", verbosity: int = 1):
    """
    Function to normalize the train data.
    Fits StandardScaler or MinMaxScaler on given train DataFrame and also outputs the fitted scaler.

    :param X_train: pd.DataFrame -- Train DataFrame
    :param method: str -- Method to scale, either 'standard' or 'minmax'
    :param verbosity: int - Level of verbosity

    :return: Scaled Train DataFrame
    """
    assert method in ["standard", "minmax"], print("method must either be 'standard' or 'minmax'")

    scaler = None
    if method == "standard":
        scaler = StandardScaler()
    if method == "minmax":
        scaler = MinMaxScaler()

    if verbosity > 0:
        print(f"Normalizing train data using method '{method}' ...")

    x_train_scaled = scaler.fit_transform(X_train)
    # Transform back to pandas DataFrame
    x_train_scaled = pd.DataFrame(x_train_scaled, columns=X_train.columns, index=X_train.index)

    return x_train_scaled, scaler


def normalize_test_data(X_test: pd.DataFrame, scaler, verbosity: int = 1) -> pd.DataFrame:
    """
    Function to normalize the test data.
    Uses already fitted StandardScaler or MinMax scaler object to transform given DataFrame.

    :param X_test: pd.DataFrame - Test DataFrame
    :param scaler: Fitted StandardScaler or MinMax Scaler object, returned from 'normalize_train_data' function
    :param verbosity: int - Level of verbosity

    :return: Scaled Test DataFrame
    """

    if verbosity > 0:
        print("Normalizing test data ...")

    # Normalize
    x_test_scaled = scaler.transform(X_test)
    # Transform back to pandas DataFrame
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=X_test.columns, index=X_test.index)

    return x_test_scaled


def feature_selection(X_train=None, X_test=None, y_train=None, quantile=0.4, verbosity=0):
    """
    Function to select feature based on either f_regression or mutual_info_regression. 
    Takes train- and test-data along with the train labels to fit the SelectKBest. 
    Also takes a number [0, 1] quantile to get all the features which provide more information than the calculated quantile. 
    Returns the train and test data with the selected features. 
    The features, which are needed for the avg spearman function are kept no matter which score they achieve.
    """
    if verbosity > 0:
        print("Feature selection...")
    
    fs = SelectKBest(score_func=f_regression, k='all')  # or mutual_info_regression
    target = list(y_train.columns)[0]
    fs.fit(X_train, y_train[target])

    # Select columns based on mask
    mask = [x >= np.quantile(fs.scores_, 0.4) for x in fs.scores_] 
    X_train_fs = X_train.loc[:, mask]
    selected_features = list(X_train_fs.columns)
    #print(f"{len(selected_features)} selected")
    
    # Do not drop the features needed for avg_spearman (the needed features start with enc_dim, tuning, scoring, model)
    sf = list(X_train.columns)
    sf = [f for f in sf if f not in selected_features or not f.startswith("enc_dim_") or not f.startswith("tuning_") or not f.startswith("scoring_") or not f.startswith("model_")]
    #print(f"After including the needed features: {len(sf)}")
    
    # Select the features and return the data frames
    X_train = X_train[sf]
    if X_test is not None:
        X_test = X_test[sf]
    
    return X_train, X_test
    