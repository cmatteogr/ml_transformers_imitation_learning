"""
"""
import numpy as np
import pandas as pd


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