"""

"""
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.supervised_baselines.utils.utils import create_activity_labels, create_sliding_windows


def preprocess(data_filepath: str,
               windows_size: int=128,
               step_size: int=64,
               test_size: float=0.2,
               method: str='heard_rate',
               threshold = 2000
               ):
    """
    Preprocess data

    :param data_filepath: data filepath
    :param windows_size: sequence length for the transformer
    :param step_size: overlap between windows
    :param test_size: evaluation dataset size
    :param method: method to find activity labels
    :param threshold: threshold to find activity labels

    :return:
    """

    # load and Prepare the Data
    print("loading data")
    df = pd.read_csv(data_filepath)
    # Let's use only the sternum sensors as features. to simplify the problem, later we can use more features
    # filter columns with 'sternum' word in them
    feature_columns = df.columns.tolist()
    print(f"using {len(feature_columns)} features: {feature_columns}")

    # use the accelerometer features to create labels
    # create Target Labels
    labels, labels_features_cols = create_activity_labels(df, threshold=threshold, method=method)
    # remove columns from df
    df.drop(columns=labels_features_cols, inplace=True)
    features_names = df.columns.tolist()

    # Apply dataset split
    X_train, X_val, y_train, y_val = train_test_split(df, labels, test_size=test_size, random_state=42, stratify=labels)

    # create sliding windows from the training set
    X_train, y_train = create_sliding_windows(X_train.values, y_train, windows_size, step_size)

    # create sliding windows from the validation set
    X_val, y_val = create_sliding_windows(X_val, y_val, windows_size, step_size)

    # apply standard normalization
    # train the model using the training data and apply to training data and validation data to avoid data leakage
    num_train_samples, _, num_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_features)
    # init standard scaler
    scaler = StandardScaler()
    scaler.fit(X_train_reshaped)
    # apply normalization and reshape
    X_train = scaler.transform(X_train_reshaped).reshape(num_train_samples, windows_size, num_features)
    # apply normalization and reshape
    num_val_samples = X_val.shape[0]
    X_val_reshaped = X_val.reshape(-1, num_features)
    X_val = scaler.transform(X_val_reshaped).reshape(num_val_samples, windows_size, num_features)

    # log parameters
    preprocess_params = {
        'window_size': windows_size,
        'step_size': step_size,
        'test_size': test_size,
        'threshold': threshold,
        'method': method,
        'features': features_names
    }
    mlflow.log_params(preprocess_params)

    # return preprocessed data
    return X_train, X_val, y_train, y_val
