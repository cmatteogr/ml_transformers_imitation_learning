"""

"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import mlflow
from src.supervised_baselines.models.attention import Seq_Transformer
import optuna


def train(X_train,
          X_val,
          y_train,
          y_val,
          window_size: int,
          batch_size: int= 64,
          num_epochs: int= 15,
          method: str='heard_rate'):
    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # convert to PyTorch Tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    # create DataLoaders, create batches
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # initialize Model, Loss, and Optimizer
    n_features = X_train.shape[2]
    # we can use other classes, for now keep it simple
    n_classes = 2  # binary class, 0 = Low activity, 1 = High activity

    # mlflow params
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_epochs", num_epochs)

    # ini the Transformer model
    # check the following post to understand Transformer architecture
    # https://poloclub.github.io/transformer-explainer/
    model = Seq_Transformer(
        n_channel=n_features,
        len_sw=window_size,
        n_classes=n_classes,
        dim=128,  # internal Transformer dimension, the embedding space, the embedding layer
        depth=4,  # number of Transformer layers, 0 to 4, chain of transformers
        heads=8,  # number of attention heads, 4 to 16
        mlp_dim=256,  # dimension of the MLP, dimension to expand to apply RELU, GELU or any other
        dropout=0.1
    )
    model.to(device)

    def train_model(trial):
        # Init the Hyperparameters to change
        learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-2, log=True)

        print("learning_rate:", learning_rate)

        # Initialize the criterion, it's a binary classification, LogLoss may work, but we will use CrossEntropyLoss
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        # to understand why CrossEntropyLoss is a good option for classification vs another loss function check the following:
        # https://www.youtube.com/watch?v=QBbC3Cjsnjg
        # https://www.youtube.com/watch?v=KHVR587oW8I
        # https://www.youtube.com/watch?v=Fv98vtitmiA
        criterion = nn.CrossEntropyLoss()
        # initialize optimizer Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        print('strat transformer training')
        best_trial_val_accuracy = float('-inf')
        # for each epoch, train the model
        for epoch in range(num_epochs):
            # set the model to training mode, method comes from pytorch nn.Module class
            # this is method is useful to activate function needed only on training phase, like Dropout
            # https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train
            model.train()
            # init the loss for the epoch
            loss_accum = 0.0
            # for each batch in the train_loader
            for i, (sequences, labels) in enumerate(train_loader):
                # move to the device
                sequences, labels = sequences.to(device), labels.to(device)
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
                    print(f'epoch [{epoch + 1}/{num_epochs}], step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

            # Calculate train batch loss
            avg_train_loss = loss_accum / len(train_loader)
            print(f'epoch [{epoch + 1}/{num_epochs}], train loss: {avg_train_loss:.2f}')

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
                    # move to device
                    sequences, labels = sequences.to(device), labels.to(device)
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
            print(f'epoch [{epoch + 1}/{num_epochs}], val accuracy: {val_accuracy:.2f}%')

            # avoid overfitting using accuracy pruned
            trial.report(val_accuracy, epoch)
            if trial.should_prune():
                print("  Trial pruned by Optuna.")
                raise optuna.exceptions.TrialPruned()

            # Track best validation loss for this trial
            best_trial_val_accuracy = max(best_trial_val_accuracy, val_accuracy)

        print(f"--- Trial {trial.number} Finished. Best Val Loss: {best_trial_val_accuracy:.6f} ---")
        return best_trial_val_accuracy  # Optuna minimizes this value

    # Execute optuna optimizer study
    print('train WildPPG Transformer')
    study_name = f"wild_ppg_transformer_{method}"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True,
                                direction='maximize')
    study.optimize(train_model, n_trials=50)
    # Get Best parameters
    best_params = study.best_params
    best_value = study.best_value
    print(f'best params: {best_params}')
    print(f'best accuracy: {best_value}')
    mlflow.log_param('transformer_params', str(best_params))

    # train final model with best hyper-parameters
    learning_rate = best_params['learning_rate']
    criterion = nn.CrossEntropyLoss()
    # initialize optimizer Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # for each epoch, train the model
    for epoch in range(num_epochs):
        # set the model to training mode, method comes from pytorch nn.Module class
        # this is method is useful to activate function needed only on training phase, like Dropout
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train
        model.train()
        # init the loss for the epoch
        loss_accum = 0.0
        # for each batch in the train_loader
        for i, (sequences, labels) in enumerate(train_loader):
            # move to the device
            sequences, labels = sequences.to(device), labels.to(device)
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
                print(f'epoch [{epoch + 1}/{num_epochs}], step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

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
                # move to device
                sequences, labels = sequences.to(device), labels.to(device)
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
        print(f'epoch [{epoch + 1}/{num_epochs}], val accuracy: {val_accuracy:.2f}%')