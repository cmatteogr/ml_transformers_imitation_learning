"""

"""
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_activity_labels(df: pd.DataFrame,
                           threshold: float=1.2,
                           method: str='acc_magnitude') -> tuple[np.ndarray, list[str]]:
    """
    Creates activity labels based on the magnitude of accelerometer data.

    :param df: The input dataframe with sensor data.
    :param threshold: The magnitude threshold to distinguish high/low activity.
    :param method: Define the method to use to create the label

    :return: An array of labels (0 for low activity, 1 for high activity).
    """
    match method:
        case 'acc_magnitude':
            labels_features_cols = ['sternum_acc_x', 'sternum_acc_y', 'sternum_acc_z']
            # calculate the vector magnitude of the accelerometer data
            acc_mag = np.sqrt(np.sum(df[labels_features_cols] ** 2, axis=1))
            # classify based on the threshold, large magnitude indicates high activity, small indicates low
            # 0 = low_activity, 1 = high_activity
            labels = (acc_mag > threshold).astype(int)
        case 'heard_rate':
            labels_features_cols = ['sternum_ecg']
            # Select the column as a Series by using its name directly
            heart_rate_series = df['sternum_ecg']
            # calculate the abd hear rate
            abs_heard_rate = np.abs(heart_rate_series)
            # classify based on the threshold, large magnitude indicates high activity, small indicates low
            # 0 = low_activity, 1 = high_activity
            labels = (abs_heard_rate > threshold).astype(int)
        case _:
            raise Exception(f"Invalid labels creation method: {method}")

    print(f"label distribution: {np.bincount(labels)} (0: low, 1: high)")
    # return label
    return labels.values, labels_features_cols

def create_sliding_windows(data: np.array, labels: np.array, window_size: int, step_size: int) -> tuple:
    """
    Creates sliding windows from time-series data.

    :param data: The array of features.
    :param labels: The array of labels.
    :param window_size: The number of time steps in each window.
    :param step_size: The step size to slide the window.

    :return: A tuple containing the windowed data and corresponding labels.

    NOTE: window_size and step_size are used to apply overlapping, it's a common practice in time-series datasets
    https://medium.com/data-science/sliding-windows-in-pandas-40b79edefa34

    """
    print(f"sliding windows, window_size: {window_size}, step_size: {step_size}" )
    # init sequences, window_labels
    sequences = []
    window_labels = []
    # generate a range list using step_size to force overlapping using window_size
    for pivot in range(0, len(data) - window_size, step_size):
        # extract a window of data
        window = data[pivot:pivot + window_size]
        # extract the label for the window
        label_window = labels[pivot:pivot + window_size]
        # get general label per window counting the labels and selecting the max value
        # np.bincount, used to count occurrences of each value in an array
        # https://numpy.org/devdocs/reference/generated/numpy.bincount.html
        # argmax, returns the indices of the maximum values along an axis
        # https://numpy.org/devdocs/reference/generated/numpy.argmax.html
        label = np.bincount(label_window).argmax()

        # append the window and label to the lists
        sequences.append(window)
        window_labels.append(label)

    # return sequences and labels
    return np.array(sequences), np.array(window_labels)

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
