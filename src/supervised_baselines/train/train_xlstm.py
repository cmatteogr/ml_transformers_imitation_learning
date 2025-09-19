import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import mlflow
import optuna

from src.supervised_baselines.models.xlstm import xLSTM


def train(X_train,
                X_val,
                y_train,
                y_val,
                batch_size: int = 64,
                num_epochs: int = 15,
                method: str = 'heart_rate'):
    """
    Trains and optimizes an xSLTM model using Optuna for hyperparameter tuning.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert numpy arrays to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # Create DataLoaders for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    n_features = X_train.shape[2]
    n_classes = 2  # Binary classification

    mlflow.log_param("model_type", "xSLTM")
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_epochs", num_epochs)

    def objective(trial):
        # Define hyperparameter search space for Optuna
        hidden_size = trial.suggest_int('hidden_size', 32, 256, log=True)
        num_layers = trial.suggest_int('num_layers', 1, 4)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)

        # Initialize the xSLTM model with trial parameters
        model = xLSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            n_classes=n_classes,
            dropout=dropout
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        print(f"\n--- Starting Trial {trial.number} ---")
        print(f"  Params: hidden_size={hidden_size}, num_layers={num_layers}, lr={learning_rate:.6f}, dropout={dropout:.2f}")

        best_trial_val_accuracy = 0.0

        for epoch in range(num_epochs):
            model.train()
            loss_accum = 0.0
            for sequences, labels in train_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                loss_accum += loss.item()

            # Validation phase
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences, labels = sequences.to(device), labels.to(device)
                    outputs = model(sequences)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = 100 * correct / total
            print(f'  Epoch [{epoch + 1}/{num_epochs}], Val Accuracy: {val_accuracy:.2f}%')

            # Report validation accuracy to Optuna for pruning
            trial.report(val_accuracy, epoch)
            if trial.should_prune():
                print("  Trial pruned by Optuna.")
                raise optuna.exceptions.TrialPruned()

            best_trial_val_accuracy = max(best_trial_val_accuracy, val_accuracy)

        print(f"--- Trial {trial.number} Finished. Best Val Accuracy: {best_trial_val_accuracy:.2f}% ---")
        return best_trial_val_accuracy

    # Create and run the Optuna study
    study_name = f"wild_ppg_xsltm_{method}"
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction='maximize'
    )
    study.optimize(objective, n_trials=50)

    # Log best parameters and results from the study
    best_params = study.best_params
    best_value = study.best_value
    print("\n--- Optuna Study Complete ---")
    print(f"Best validation accuracy: {best_value:.2f}%")
    print(f"Best parameters: {best_params}")
    mlflow.log_params(best_params)
    mlflow.log_metric("best_val_accuracy", best_value)

    # --- Train the final model with the best hyperparameters ---
    print("\n--- Training Final Model with Best Parameters ---")
    final_model = xLSTM(
        input_size=n_features,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        n_classes=n_classes,
        dropout=best_params['dropout']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])

    for epoch in range(num_epochs):
        final_model.train()
        train_loss = 0.0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = final_model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        mlflow.log_metric('final/train_loss', avg_train_loss, step=epoch)

        # Final validation
        final_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = final_model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        mlflow.log_metric('final/val_accuracy', val_accuracy, step=epoch)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Final Train Loss: {avg_train_loss:.4f}, Final Val Accuracy: {val_accuracy:.2f}%')

