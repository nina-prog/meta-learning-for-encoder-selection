import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
import tensorflow_ranking as tfr

import src.evaluate_regression as er
from src.feature_engineering import normalize_train_data, normalize_test_data

# Set tensorflow seed
tf.random.set_seed(42)

# Global Variables
EPOCHS: int
BATCH_SIZE: int
DROP_OUT: float
LEARNING_RATE: float
NORMALIZATION_METHOD: str


def init_global_vars(cfg) -> None:
    """ Function to initialize global variables

    :param cfg: dict -- Parsed config file
    :return: None
    """
    global EPOCHS, BATCH_SIZE, DROP_OUT, LEARNING_RATE, NORMALIZATION_METHOD

    EPOCHS = cfg["neural_net"]["epochs"]
    BATCH_SIZE = cfg["neural_net"]["batch_size"]
    DROP_OUT = cfg["neural_net"]["dropout_rate"]
    LEARNING_RATE = cfg["neural_net"]["learning_rate"]
    NORMALIZATION_METHOD = cfg["feature_engineering"]["normalize"]["method"]


def get_model(input_dim: int, output_dim: int, drop_out: float, lr: float) -> tf.keras.Sequential:
    """ Returns compiled model

    :param input_dim: int -- Input Dimension of the Data
    :param output_dim: int -- Number of dimensions of the y_train, i.e. # classes
    :param drop_out: float -- Drop out rate
    :param lr: float -- Learning rate for adam optimizer
    :return: model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=input_dim),
        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dropout(rate=drop_out, seed=42),
        tf.keras.layers.Dense(units=1024, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dropout(rate=drop_out, seed=42),
        tf.keras.layers.Dense(units=output_dim)
    ])

    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=3.0)

    # Compile model with LISTMLELoss and adam optimizer
    model.compile(optimizer=optimizer,
                  loss=tfr.keras.losses.ListMLELoss())

    return model


def get_callbacks() -> list:
    """ Returns Callbacks in form of a list

    :return: list of tf.keras.callbacks
    """

    # Callbacks for early stopping and reduce LR
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=20,  # Number of epochs without improvement
        restore_best_weights=True,
        verbose=0
    )

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,  # Reduce the learning rate by half when the loss plateaus
        patience=15,  # Number of epochs with no improvement after which the learning rate will be reduced
        min_lr=1e-6,  # Lower bound on the learning rate
        verbose=0
    )

    return [early_stopping_callback, reduce_lr_callback]


def perform_cv_neural_net(cfg, cv_indices, X_train, y_train, X_train_org, factors, new_index, verbosity):

    # Init global vars such as EPOCHS, ...
    init_global_vars(cfg)

    scores = []
    histories = {}

    if verbosity > 0:
        print(f"Performing CV ... ")

    for i, fold in enumerate(cv_indices):
        # Save original schema to build df_pred for custom scorer
        X_train_org_te = X_train_org.iloc[fold[1]]

        # Get Data using the custom cv indices
        X_tr = X_train.iloc[fold[0]]
        X_te = X_train.iloc[fold[1]]
        y_tr = y_train.iloc[fold[0]]
        y_te = y_train.iloc[fold[1]]

        # Normalize
        X_tr, scaler = normalize_train_data(X_train=X_tr, method=NORMALIZATION_METHOD, verbosity=verbosity)
        X_te = normalize_test_data(X_test=X_te, scaler=scaler, verbosity=verbosity)

        # Get model and fit using the curren train data
        model = get_model(input_dim=X_tr.shape[1], output_dim=y_tr.shape[1], drop_out=DROP_OUT, lr=LEARNING_RATE)
        history = model.fit(x=X_tr, y=y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
                            callbacks=get_callbacks())
        histories[f"{i}-fold"] = history

        # Make prediction on current test data
        y_pred = pd.DataFrame(model.predict(X_te), columns=y_tr.columns, index=X_te.index)
        df_pred = pd.merge(
            pd.concat([X_train_org_te, y_te], axis=1).melt(id_vars=factors, value_name="rank").dropna(axis=0),
            pd.concat([X_train_org_te, y_pred], axis=1).melt(id_vars=factors, value_name="rank_pred"),
            on=factors + ["encoder"], how="left")

        rankings_test = er.get_rankings(df_pred, factors=factors, new_index=new_index, target="rank")
        rankings_pred = er.get_rankings(df_pred, factors=factors, new_index=new_index, target="rank_pred")

        # Custom scorer
        avg_spearman = er.average_spearman(rankings_test, rankings_pred)
        scores.append(avg_spearman)
        if verbosity > 0:
            print(f"\n AVG Spearman of Fold {i}: {avg_spearman}\n")

    if verbosity > 0:
        print(f"\nAverage Spearman of all folds: {np.mean(scores):.4f} +/- {np.std(scores):.4f}\n")

    return scores, histories


def fit_neural_net(cfg, X_train, y_train, X_test, verbosity):

    # Init global vars such as EPOCHS, ...
    init_global_vars(cfg)

    # Normalize
    X_train, scaler = normalize_train_data(X_train=X_train, method=NORMALIZATION_METHOD, verbosity=verbosity)
    X_test = normalize_test_data(X_test=X_test, scaler=scaler, verbosity=verbosity)

    # For debugging reasons
    X_train.to_csv("X_train.csv")
    X_test.to_csv("X_test.csv")
    y_train.to_csv("y_train.csv")

    # Get model and fit using the curren train data
    if verbosity > 0:
        print(f"Fitting neural net on whole train data ...")

    model = get_model(input_dim=X_train.shape[1], output_dim=y_train.shape[1], drop_out=DROP_OUT, lr=LEARNING_RATE)
    history = model.fit(x=X_train, y=y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
                        callbacks=get_callbacks())

    return model, history


def make_prediction(cfg: dict, model: tf.keras.Sequential, X_test: pd.DataFrame,
                    columns: list, verbosity: int) -> None:
    """ Function to make final prediction and save results in data/predictions

    :param cfg: dict -- Parsed config file
    :param model: tf.keras.Sequential -- Fitted Model yielded from fit_neural_net() func
    :param X_test: pd.DataFrame -- Holdout Test set
    :param columns: list -- List of column names for the prediction
    :param verbosity: int -- Level of Verbosity
    :return: None
    """

    if verbosity > 0:
        print("Making final prediction on holdout-set X_test ...")

    # Make prediction on current test data
    y_pred = pd.DataFrame(model.predict(X_test, verbose=0), columns=columns, index=X_test.index)

    # Save prediction
    y_pred.to_csv(cfg["paths"]["result_path"], index=False)
    if verbosity > 0:
        print(f"Saved final prediction in '{cfg['paths']['result_path']}'")


def plot_cv_results(histories: dict, scores: list) -> None:
    """ Plots results of the CV

    :param histories: dict -- Dict of history objects
    :param scores: list -- List of Average Spearman scores
    :return: None
    """

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    for key, item in histories.items():
        loss = item.history["loss"]
        sns.lineplot(x=np.arange(1, len(loss) + 1), y=loss, label=key, ax=axes[0])
    axes[0].legend()
    axes[0].set_title("Loss in each Epoch and Fold")
    axes[0].set_xlabel("# Epoch")
    axes[0].set_ylabel("Loss")
    # Second plot
    sns.barplot(x=np.arange(0, len(scores)), y=scores, ax=axes[1], edgecolor="black")
    axes[1].set_title(f"Average Spearman in each fold on holdout-set" +
                      f" ({np.mean(scores):.4f} +/- {np.std(scores):.4f})")
    axes[1].set_xlabel("k-fold")
    axes[1].set_ylabel("Average Spearman")
    axes[1].axhline(y=np.mean(scores), linestyle="--", color="red")
    plt.show()
