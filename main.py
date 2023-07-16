"""
Module for the pipeline.
Execute via python3 main.py --config configs/config.yaml
or if you want to run pipeline with subsample of size 100
python3 main.py --config configs/config.yaml --subsample 100
"""
import time
import pandas as pd

import src.utils
import src.load_datasets

#import src.regression_method
import src.pointwise_method
import src.pairwise_method


def main():
    """
    Function that executes the pipeline
    """

    # Track time for execution
    start_time = time.time()
    print(30 * "=" + " PIPELINE STARTED " + 30 * "=")

    # Parse arguments and read config file cfg
    args = src.utils.parse_args()
    cfg, cfg_path = src.utils.parse_config(args)
    verbosity = cfg["general"]["verbosity"]
    
    ### LOAD DATA ###
    df_train = src.load_datasets.load_dataset(path=cfg["paths"]["train_rank_data_path"],
                                              verbosity=verbosity,
                                              subsample=args.subsample)
    
    # Load Test data (Hold-out-set) that is used for course-internal scoring
    X_test = src.load_datasets.load_test_data(path=cfg["paths"]["test_values_path"],
                                              verbosity=verbosity,
                                              subsample=args.subsample)
    
    
    # Check method to use
    # Valid arguments are 'regression', 'pointwise', 'pairwise', 'listwise'
    # For each method preprocess data, train a model and predict results
    if cfg["general"]["method"] == "regression":
        # ToDo
        pass
    elif cfg["general"]["method"] == "pointwise":
        # Constants for supervisors functions
        FACTORS = ["dataset", "model", "tuning", "scoring"]
        NEW_INDEX = "encoder"
        TARGET = "rank"
        
        # Preprocess data
        X_train, y_train, X_test = src.pointwise_method.preprocessing(df_train=df_train, 
                                                                      X_test=X_test, 
                                                                      config=cfg)
        
        # Get indices for custom cross validation that is used within the modelling.py module
        indices = src.evaluate_regression.custom_cross_validated_indices(df_train,
                                                                         FACTORS, 
                                                                         TARGET,
                                                                         n_splits=cfg["modelling"]["k_fold"],
                                                                         shuffle=True, 
                                                                         random_state=1444)
        
        # Train model and predict results of test data
        model = src.pointwise_method.modelling(X_train=X_train,
                                               y_train=y_train,
                                               indices=indices,
                                               config_path=cfg_path,
                                               config=cfg)
        src.pointwise_method.prediction_pointwise(model=model,
                                                  X_test=X_test,
                                                  config=cfg)
    elif cfg["general"]["method"] == "pairwise":
        # Constants for supervisors functions
        FACTORS = ["dataset", "model", "tuning", "scoring"]
        NEW_INDEX = "encoder"
        TARGET = "rank"
        
        # Preprocess data
        X_train, y_train, X_test, base_df = src.pairwise_method.preprocessing(df_train=df_train, 
                                                                               X_test=X_test, 
                                                                               config=cfg)
        
        # Get indices for custom cross validation that is used within the modelling.py module
        indices = src.evaluate_regression.custom_cross_validated_indices(pd.concat([X_train, y_train], axis=1),
                                                                         list(X_train.columns),  # factors
                                                                         list(y_train.columns),  # target
                                                                         n_splits=cfg["modelling"]["k_fold"],
                                                                         shuffle=True, 
                                                                         random_state=1444)
        
        # Train model and predict results of test data
        model = src.pairwise_method.modelling(X_train=X_train,
                                               y_train=y_train,
                                               base_df=base_df,
                                               indices=indices,
                                               config_path=cfg_path,
                                               config=cfg)
        
        merge_cols = list(base_df)
        merge_cols.remove("rank")
        merge_cols.remove("encoder")
        src.pairwise_method.prediction_pointwise(model=model,
                                                 X_test=X_test,
                                                 target_columns=list(y_train.columns),
                                                 merge_cols=merge_cols,
                                                 config=cfg)
    elif cfg["general"]["method"] == "listwise":
        pass
    else:
        print(f"'{cfg['general']['method']}' is no valid option. ")
        print(f"End run!")
    
    
    # Track time for total runtime and display end of pipeline
    runtime = time.time() - start_time
    print(30 * "=" + f" PIPELINE FINISHED ({src.utils.display_runtime(runtime)}) " + 30 * "=")


if __name__ == "__main__":
    main()
