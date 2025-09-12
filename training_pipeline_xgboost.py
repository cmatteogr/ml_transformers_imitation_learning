import mlflow
from src.supervised_baselines.data_preprocess.preprocess_xgboost import preprocess
from src.supervised_baselines.train.train_xgboost import train
from src.supervised_baselines.utils.constants import MLFLOW_HOST, MLFLOW_PORT

METHOD_LABELS = 'heard_rate'
THRESHOLD_LABELS = 2000
#METHOD_LABELS = 'acc_magnitude'
#THRESHOLD_LABELS = 1.1

# initialize MLFLow
mlflow.set_tracking_uri(uri=f"http://{MLFLOW_HOST}:{MLFLOW_PORT}")
# set experiment name
mlflow.set_experiment(f"wild_ppg_classification_xgboost_{METHOD_LABELS}_running_activity")

# init variables
data_filepath = './data/WildPPG_data.csv'

NUM_EPOCHS = 10

print("run training pipeline")
# Start an MLFlow run context to log parameters, metrics, and artifacts
with mlflow.start_run() as run:
    # apply preprocessing
    X_train, X_val, y_train, y_val = preprocess(data_filepath,
                                                method=METHOD_LABELS,
                                                threshold=THRESHOLD_LABELS)

    # train transformer
    train(X_train,
          X_val,
          y_train,
          y_val,
          method=METHOD_LABELS)
