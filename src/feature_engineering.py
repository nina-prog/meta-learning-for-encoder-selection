import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
