"""

"""
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from src.supervised_baselines.utils.utils import create_activity_labels, create_sliding_windows


def preprocess(data_filepath: str,
               test_size: float=0.2,
               method: str='heard_rate',
               threshold = 2000
               ):
    """
    Preprocess data

    :param data_filepath: data filepath
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


    preprocess_params = {
        'test_size': test_size,
        'threshold': threshold,
        'method': method,
        'features': features_names
    }
    mlflow.log_params(preprocess_params)

    # return preprocessed data
    return X_train, X_val, y_train, y_val
