import mlflow
from src.supervised_baselines.data_preprocess.preprocess import preprocess
from src.supervised_baselines.train.train_transformer import train

METHOD_LABELS = 'heard_rate'
THRESHOLD_LABELS = 2000
#METHOD_LABELS = 'acc_magnitude'
#THRESHOLD_LABELS = 1.1

# initialize MLFLow
MLFLOW_HOST='127.0.0.1'
MLFLOW_PORT=5000
mlflow.set_tracking_uri(uri=f"http://{MLFLOW_HOST}:{MLFLOW_PORT}")
# set experiment name
mlflow.set_experiment(f"wild_ppg_transformer_classification_{METHOD_LABELS}_running_activity")

# init variables
data_filepath = './data/WildPPG_data.csv'
WINDOW_SIZE = 128      # sequence length for the transformer
STEP_SIZE = 64         # overlap between windows

NUM_EPOCHS = 10

# Start an MLFlow run context to log parameters, metrics, and artifacts
with mlflow.start_run() as run:
    # apply preprocessing
    X_train, X_val, y_train, y_val = preprocess(data_filepath,
                                                windows_size=WINDOW_SIZE,
                                                step_size=STEP_SIZE,
                                                method=METHOD_LABELS,
                                                threshold=THRESHOLD_LABELS)

    # train transformer
    train(X_train,
          X_val,
          y_train,
          y_val,
          window_size=WINDOW_SIZE,
          num_epochs=NUM_EPOCHS,
          method=METHOD_LABELS)
