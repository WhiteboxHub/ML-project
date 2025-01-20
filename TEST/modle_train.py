import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pytorch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")  # Replace with your MLflow server URI if different

# Load dataset
dataset = pd.read_csv("creditcard_2023.csv")

# Subset of the dataset
dataset_subset = dataset  # Use a subset for quicker iteration

# Feature and label separation
ip = dataset_subset.drop("Class", axis=1).values  # Ensure "Class" is the correct column name
op = dataset_subset["Class"].values  # Match the case of the label column
x_train, x_test, y_train, y_test = train_test_split(ip, op, test_size=0.1, random_state=42)

x = x_test
y = y_test
# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(x)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Add dimension for binary output
X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the PyTorch model
class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetectionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# Training function
def train_model(epochs, learning_rate, batch_size=32):
    input_dim = X_train_tensor.shape[1]  # Get the number of features
    model = FraudDetectionModel(input_dim)

    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Start MLflow run
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("model_architecture", "3-layer fully connected NN")
        mlflow.log_param("activation_function", "ReLU")
        mlflow.log_param("loss_function", "Binary Cross-Entropy")
        mlflow.log_param("optimizer", "Adam")

        # Training loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Log epoch loss
            avg_loss = running_loss / len(train_loader)
            mlflow.log_metric("training_loss", avg_loss, step=epoch)

            # Evaluate on validation set (optional)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(test_loader)
            mlflow.log_metric("validation_loss", avg_val_loss, step=epoch)

            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Evaluate the model
        correct = 0
        total = 0
        y_pred, y_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                predictions = (outputs > 0.5).float()
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)
                y_pred.extend(predictions.numpy())
                y_true.extend(y_batch.numpy())

        accuracy = correct / total
        mlflow.log_metric("accuracy", accuracy)
        print(f"Model Accuracy: {accuracy:.4f}")

        # Compute additional metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc", auc)

        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")

        # Save confusion matrix as an artifact
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
        cm_df.to_csv("confusion_matrix.csv")
        mlflow.log_artifact("confusion_matrix.csv")

        # Log the model with input example
        example_input = X_test_tensor[:1].numpy()  # Convert to numpy array
        mlflow.pytorch.log_model(
            model, 
            "pytorch_model",
            input_example=example_input
        )
        print("Model logged to MLflow successfully.")

        # Save model weights
        torch.save(model.state_dict(), "model_weights.pth")
        mlflow.log_artifact("model_weights.pth")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        model_name = "fraud_detection_model"
        # Register the model in the registry
        mlflow.register_model(model_uri, model_name)
# Train the model with different hyperparameters
train_model(epochs=20, learning_rate=1.001)
# small learning rate
train_model(epochs=30, learning_rate=1e-4)

# large learning rate
train_model(epochs=30, learning_rate=1e-1)

train_model(epochs=30, learning_rate=0.0005)
