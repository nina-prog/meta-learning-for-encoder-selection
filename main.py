"""
Module for the pipeline.
Execute via python3 main.py --config configs/config.yaml
or if you want to run pipeline with subsample of size 100
python3 main.py --config configs/config.yaml --subsample 100
"""
import time
import pandas as pd

import mlflow.sklearn

import src.utils
import src.load_datasets
import src.modelling
import src.mlflow_registry
import src.encoding
import src.evaluate_regression


def main():
    """
    Function that executes the pipeline
    """

    # Track time for execution
    start_time = time.time()
    print(20 * "=" + " PIPELINE STARTED " + 20 * "=")

    # Parse arguments and read config file cfg
    args = src.utils.parse_args()
    cfg, cfg_path = src.utils.parse_config(args)
    verbosity = cfg["general"]["verbosity"]

    # Load Data
    #X_train, y_train = src.load_datasets.load_train_data(path=cfg["paths"]["train_data_path"],
    #                                                     verbosity=verbosity,
    #                                                     subsample=args.subsample)
    #X_test = src.load_datasets.load_test_data(path=cfg["paths"]["test_values_path"],
    #                                          verbosity=verbosity,
    #                                          subsample=args.subsample)

    factors = ["dataset", "model", "tuning", "scoring"]
    new_index = "encoder"
    target = "cv_score"

    df_train = src.load_datasets.load_dataset(path=cfg["paths"]["train_data_path"])
    X_train, X_test, y_train, y_test = src.evaluate_regression.custom_train_test_split(df_train, factors, target)
    X_test_original = X_test.copy()

    # General encodings: One Hot Encode (OHE) subset of features
    X_train, ohe = src.encoding.ohe_encode_train_data(X_train=X_train,
                                                      cols_to_encode=cfg["feature_engineering"]["features_to_ohe"],
                                                      verbosity=verbosity)
    X_test = src.encoding.ohe_encode_test_data(X_test=X_test,
                                               cols_to_encode=cfg["feature_engineering"]["features_to_ohe"],
                                               ohe=ohe, verbosity=verbosity)

    # Encoder encoding: Poincare Embeddings for feature "encoder"
    X_train, poincare_model = src.encoding.poincare_encoding(path_to_graph=cfg["paths"]["graph_path"], data=X_train,
                                                             column_to_encode="encoder",
                                                             encode_dim=
                                                             cfg["feature_engineering"]["poincare_embedding"]["dim"],
                                                             explode_dim=
                                                             cfg["feature_engineering"]["poincare_embedding"][
                                                                 "explode_dim"],
                                                             epochs=cfg["feature_engineering"]["poincare_embedding"][
                                                                 "epochs"],
                                                             verbosity=verbosity)
    X_test, poincare_model = src.encoding.poincare_encoding(path_to_graph=cfg["paths"]["graph_path"], data=X_test,
                                                            column_to_encode="encoder",
                                                            encode_dim=cfg["feature_engineering"]["poincare_embedding"][
                                                                "dim"],
                                                            explode_dim=
                                                            cfg["feature_engineering"]["poincare_embedding"][
                                                                "explode_dim"],
                                                            epochs=cfg["feature_engineering"]["poincare_embedding"][
                                                                "epochs"],
                                                            verbosity=verbosity)

    X_train.to_csv("data/preprocessed/X_train.csv", index=False)
    X_test.to_csv("data/preprocessed/X_test.csv", index=False)
    y_train.to_csv("data/preprocessed/y_train.csv", index=False)

    # Log model evaluation to mlflow registry
    mlflow.sklearn.autolog(log_models=False)
    with mlflow.start_run(tags=src.mlflow_registry.get_mlflow_tags(X_train, cfg)) as run:
        # Perform CV and train model
        model, cv_result = src.modelling.train_model(model=cfg["modelling"]["model"],
                                                     train_data=X_train,
                                                     train_labels=y_train,
                                                     scoring=cfg["modelling"]["scoring"],
                                                     hyperparam_grid=None,
                                                     verbosity=verbosity,
                                                     k_fold=cfg["modelling"]["k_fold"])
        # Log additional information to mlflow run
        src.mlflow_registry.log_model_eval(cv_result, cfg, cfg_path, run, verbosity)

        # Make final predictions on test data
        y_pred = src.modelling.make_prediction(model=model, test_data=X_test,
                                               result_path=cfg["paths"]["result_path"], verbosity=verbosity)

        # Concat to df_pred for spearman evaluation
        df_pred = pd.concat([X_test_original, y_test, y_pred], axis=1)
        df_pred.to_csv("data/preprocessed/df_pred.csv", index=False)

        rankings_test = src.evaluate_regression.get_rankings(df_pred, factors=factors,
                                                             new_index=new_index, target="cv_score")
        rankings_pred = src.evaluate_regression.get_rankings(df_pred, factors=factors,
                                                             new_index=new_index, target="cv_score_pred")
        # Get Average Spearman, print and log it
        avg_spearman = src.evaluate_regression.average_spearman(rankings_test, rankings_pred)
        print(f"Average Spearman: {avg_spearman:.4f}")
        mlflow.log_metric(key="average_spearman", value=avg_spearman)

    # Track time for total runtime and display end of pipeline
    runtime = time.time() - start_time
    print(20 * "=" + f" PIPELINE FINISHED ({src.utils.display_runtime(runtime)}) " + 20 * "=")


if __name__ == "__main__":
    main()
