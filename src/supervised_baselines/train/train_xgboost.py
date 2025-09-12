"""

"""
import xgboost as xgb
import optuna
import mlflow
from sklearn.metrics import accuracy_score


def train(X_train,
          X_val,
          y_train,
          y_val,
          method: str='heard_rate'):


    # define the Objective Function for Optuna
    def train_model(trial):
        # define the hyperparameter search space
        param = {
            'objective': 'binary:logistic',
            'device':'cuda',
            'eval_metric': 'logloss',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.2, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
        }

        # add booster-specific parameters
        if param['booster'] == 'gbtree' or param['booster'] == 'dart':
            param['max_depth'] = trial.suggest_int('max_depth', 3, 9)
            param['eta'] = trial.suggest_float('eta', 1e-8, 1.0, log=True)
            param['gamma'] = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
            param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        if param['booster'] == 'dart':
            param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
            param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            param['rate_drop'] = trial.suggest_float('rate_drop', 1e-8, 1.0, log=True)
            param['skip_drop'] = trial.suggest_float('skip_drop', 1e-8, 1.0, log=True)

        # initialize and train the XGBoost model
        model = xgb.XGBClassifier(**param)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # make predictions and calculate accuracy
        preds = model.predict(X_val)
        accuracy = accuracy_score(y_val, preds)

        return accuracy

    # Execute optuna optimizer study
    print('train WildPPG XGBoost')
    study_name = f"wild_ppg_xgboost_{method}"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True,
                                direction='maximize')

    # start the optimization process
    study.optimize(train_model, n_trials=100)

    # print and use the results
    print(f"best trial's accuracy: {study.best_value:.4f}")
    print("best hyperparameters found:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # train the final model with the best hyperparameters
    print("training final model with the best hyperparameters...")
    best_params = study.best_params
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)

    # evaluate the final model on the test set
    test_preds = final_model.predict(X_val)
    final_accuracy = accuracy_score(y_val, test_preds)

    mlflow.log_metric('eval/val_accuracy', final_accuracy)

    print(f"\nfinal model accuracy on validation set: {final_accuracy:.4f}")