import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from src.supervised_baselines.models.attention import Seq_Transformer


def create_activity_labels(df: pd.DataFrame, acc_cols: list[str], threshold: float=1.2) -> np.ndarray:
    """
    Creates activity labels based on the magnitude of accelerometer data.

    :param df: The input dataframe with sensor data.
    :param acc_cols: A list of the three accelerometer column names (x, y, z).
    :param threshold: The magnitude threshold to distinguish high/low activity.

    :return: An array of labels (0 for low activity, 1 for high activity).
    """
    # calculate the vector magnitude of the accelerometer data
    acc_mag = np.sqrt(np.sum(df[acc_cols] ** 2, axis=1))

    # classify based on the threshold, large magnitude indicates high activity, small indicates low
    # 0 = low_activity, 1 = high_activity
    labels = (acc_mag > threshold).astype(int)

    print(f"label distribution: {np.bincount(labels)} (0: low, 1: high)")
    # return label
    return labels.values


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


# load and Prepare the Data
print("loading data")
df = pd.read_csv('./data/WildPPG_data.csv')

# Let's use only the sternum sensors as features. to simplify the problem, later we can use more features
# filter columns with 'sternum' word in them
feature_columns = [col for col in df.columns if 'sternum' in col]
print(f"using {len(feature_columns)} features: {feature_columns}")

# use the accelerometer features to create labels
# create Target Labels
# NOTE: The label generated using the same features generate Data Leakage, For now, we will keep this version
# and focus on the Transformer execution. Check the following post to know how to identify and deal with data leakage
# https://www.ibm.com/think/topics/data-leakage-machine-learning
accel_cols = ['sternum_acc_x', 'sternum_acc_y', 'sternum_acc_z']
labels = create_activity_labels(df, accel_cols, threshold=1.1)

# preprocess the data
# use StandardScaler to reduce the impact of outliers in each feature based on the Data profiling analysis
# The normalization is needed for neural networks due it's nature
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[feature_columns])

# create sliding Windows applying overlapping
WINDOW_SIZE = 128      # sequence length for the transformer
STEP_SIZE = 64         # overlap between windows
X, y = create_sliding_windows(scaled_features, labels, WINDOW_SIZE, STEP_SIZE)

print(f"num sequences created {len(X)}.")
print(f"shape of X: {X.shape} -> (num_sequences, window_size, num_features)")
print(f"shape of y: {y.shape} -> (num_sequences,)")

# split into Training and Validation Sets
# here there is another data leakage issue, because the normalization was done using the entire dataset
# the distribution of both, train and test datasets features were used to calculate the mean and variance
# to normalize the dataset, when both distributions are used information is shared
# NOTE: Part of this code was generated using Gemini, Keep in mind the LLMs are probability models, if in the
# dataset used the most likely option was allow data leakage, then probably the same will happen in the code generated
# Data Leakage could generate optimistic results, it's problem when you deploy and model uses real unseen data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# convert to PyTorch Tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
# create DataLoaders, create batches
BATCH_SIZE = 64
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# initialize Model, Loss, and Optimizer
N_FEATURES = len(feature_columns)
# we can use other classes, for now keep it simple
N_CLASSES = 2  # binary class, 0 = Low activity, 1 = High activity

# ini the Transformer model
# check the following post to understand Transformer architecture
# https://poloclub.github.io/transformer-explainer/
model = Seq_Transformer(
    n_channel=N_FEATURES,
    len_sw=WINDOW_SIZE,
    n_classes=N_CLASSES,
    dim=128,                    # internal Transformer dimension, the embedding space, the embedding layer
    depth=4,                    # number of Transformer layers, 0 to 4, chain of transformers
    heads=8,                    # number of attention heads, 4 to 16
    mlp_dim=256,                # dimension of the MLP, dimension to expand to apply RELU, GELU or any other
    dropout=0.1
)
# Initialize the criterion, it's a binary classification, LogLoss may work, but we will use CrossEntropyLoss
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
# to understand why CrossEntropyLoss is a good option for classification vs another loss function check the following:
# https://www.youtube.com/watch?v=QBbC3Cjsnjg
# https://www.youtube.com/watch?v=KHVR587oW8I
# https://www.youtube.com/watch?v=Fv98vtitmiA
criterion = nn.CrossEntropyLoss()
# initialize optimizer Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# initialize MLFLow
MLFLOW_HOST='127.0.0.1'
MLFLOW_PORT=5000
mlflow.set_tracking_uri(uri=f"http://{MLFLOW_HOST}:{MLFLOW_PORT}")
# set experiment name
mlflow.set_experiment("wild_ppg_transformer_classification_running_activity")

# Start an MLFlow run context to log parameters, metrics, and artifacts
with mlflow.start_run() as run:
    # training loop
    print('strat transformer training')
    NUM_EPOCHS = 5
    # for each epoch, train the model
    for epoch in range(NUM_EPOCHS):
        # set the model to training mode, method comes from pytorch nn.Module class
        # this is method is useful to activate function needed only on training phase, like Dropout
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train
        model.train()
        # init the loss for the epoch
        loss_accum = 0.0
        # for each batch in the train_loader
        for i, (sequences, labels) in enumerate(train_loader):
            # set to zero the gradients
            # https://docs.pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
            optimizer.zero_grad()

            # the model's forward pass expects the classification token to be handled inside
            # it returns the processed classification token
            class_token_output = model(sequences)

            # the classifier head is separate in this architecture
            # we apply it to the processed token
            outputs = model.classifier(class_token_output)

            # get the loss for the batch, output-labels
            loss = criterion(outputs, labels)
            # calculate the gradient
            # https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            loss.backward()
            # apply back propagation
            # https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html
            optimizer.step()

            # sum the batch error
            loss_accum += loss.item()
            # print every 50 batches
            if (i + 1) % 50 == 0:
                print(f'epoch [{epoch + 1}/{NUM_EPOCHS}], step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Calculate train batch loss
        avg_train_loss = loss_accum / len(train_loader)
        mlflow.log_metric('train/avg_train_loss', avg_train_loss, step=epoch)

        # validation
        # validation is in another block to avoid overfitting
        # set the model on evaluation mode, eval disable the dropout layers and normalization layers
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval
        model.eval()
        # init the correct predictions and total predictions to calculate the accuracy
        correct = 0
        total = 0
        # no Gradient initialization because we don't apply back propagation here
        with torch.no_grad():
            for sequences, labels in val_loader:
                # use the model in eval mode, get classification token
                class_token_output = model(sequences)
                # apply the classifier head, last layer, same that
                outputs = model.classifier(class_token_output)
                # get the predicted class by batch
                # https://docs.pytorch.org/docs/stable/generated/torch.max.html
                _, predicted = torch.max(outputs.data, 1)
                # get the total and correct predictions
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate validation accuracy
        val_accuracy = 100 * correct / total
        mlflow.log_metric('eval/val_accuracy', val_accuracy, step=epoch)
        print(f'epoch [{epoch + 1}/{NUM_EPOCHS}], val accuracy: {val_accuracy:.2f}%')
