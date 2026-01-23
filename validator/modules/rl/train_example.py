from __future__ import annotations
import os
import time
import numpy as np
import torch
import torch.nn as nn
import requests
import json
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict

FED_LEDGER_BASE_URL = "https://fed-ledger-prod.flock.io/api/v1"
FLOCK_API_KEY = os.environ.get("FLOCK_API_KEY")

class MLP(nn.Module):
    """Simple Multi-Layer Perceptron"""

    def __init__(self, in_dim: int, out_dim: int, hidden=(256, 256), act=nn.ReLU):
        super().__init__()
        layers = []
        last_dim = in_dim

        for h in hidden:
            layers.append(nn.Linear(last_dim, h))
            layers.append(act())
            last_dim = h

        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleDataset(Dataset):
    """Simple dataset wrapper for numpy arrays"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(data_dir: str = "data"):
    """
    Load training data from .npy files

    Args:
        data_dir: Directory containing the data files

    Returns:
        X_train, Info_train
    """
    print(f"Loading data from {data_dir}...")

    data = np.load(os.path.join(data_dir, "train.npz"))
    X_train, Info_train = data['X'], data['Info']

    print(f"X_train shape: {X_train.shape}")
    print(f"Info_train shape: {Info_train.shape}")

    return X_train, Info_train


def prepare_labels(Info: np.ndarray) -> np.ndarray:
    """
    Create labels from Info array.
    For this example, we'll create a dummy target with the same number of dimensions
    as the expected action space (V venues).
    
    The environment infers V from Info columns: V = (cols - 3) // 4
    """
    # Infer V (number of venues)
    m = Info.shape[1]
    start = 3
    V = (m - start) // 4
    
    # Create dummy targets of shape (N, V)
    labels = Info[:, start:start+V].astype(np.float32) 
    
    return labels


def train_mlp(
    data_dir: str = "data",
    output_dir: str = "runs",
    model_name: str = "mlp_example",
    hidden: Tuple[int, ...] = (256, 256),
    batch_size: int = 128,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    validation_split: float = 0.2,
    device: str = None,
    seed: int = 42,
) -> Dict:
    """
    Train a MLP model

    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save model outputs
        model_name: Name for the saved model
        hidden: Hidden layer dimensions
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        validation_split: Fraction of training data to use for validation (0.0-1.0)
        device: Device to train on (cuda/cpu)
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing training logs
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    X_all, Info_all = load_data(data_dir)

    # Prepare labels
    y_all = prepare_labels(Info_all)

    # Split data into train and validation sets
    n_samples = len(X_all)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    val_size = int(n_samples * validation_split)
    train_size = n_samples - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    X_train = X_all[train_indices]
    y_train = y_all[train_indices]
    X_val = X_all[val_indices]
    y_val = y_all[val_indices]

    print("\nData split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Create datasets and dataloaders
    train_dataset = SimpleDataset(X_train, y_train)
    val_dataset = SimpleDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Create model
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = MLP(input_dim, output_dim, hidden=hidden).to(device)

    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    # Training logs
    log = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_time": [],
    }

    print("\nStarting training...")
    total_start = time.time()

    # Training loop
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        epoch_time = time.time() - epoch_start

        # Log metrics
        log["epoch"].append(epoch)
        log["train_loss"].append(avg_train_loss)
        log["val_loss"].append(avg_val_loss)
        log["train_time"].append(epoch_time)

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[Epoch {epoch:4d}/{epochs}] "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {avg_val_loss:.6f} | "
                f"Time: {epoch_time:.2f}s"
            )

    total_time = time.time() - total_start
    print(f"\nTraining completed in {total_time:.2f}s")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save PyTorch model
    pt_path = os.path.join(output_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"Saved PyTorch model to {pt_path}")


    # Convert to ONNX
    print("\nConverting model to ONNX format...")
    model.eval()
    dummy_input = torch.randn(1, input_dim).to(device)
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
        do_constant_folding=True,
    )
    print(f"Saved ONNX model to {onnx_path}")

    # Save training log
    log_path = os.path.join(output_dir, f"{model_name}_log.npz")
    np.savez(log_path, **log)
    print(f"Saved training log to {log_path}")

    return log


def upload_to_huggingface(
    model_path: str,
    repo_id: str,
    token: str = None,
    commit_message: str = "Upload model",
):
    """
    Upload model to Hugging Face Hub

    Args:
        model_path: Path to the model file to upload
        repo_id: Hugging Face repository ID (e.g., "username/model-name")
        token: Hugging Face API token (or set HF_TOKEN environment variable)
        commit_message: Commit message for the upload
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "Error: huggingface_hub not installed. Install with: pip install huggingface_hub"
        )
        return

    print("\nUploading model to Hugging Face...")
    print(f"Repository: {repo_id}")
    print(f"Model path: {model_path}")

    # Get token from environment if not provided
    if token is None:
        token = os.environ.get("HF_TOKEN")
        if token is None:
            print(
                "Error: No Hugging Face token provided. Set HF_TOKEN environment variable or pass token parameter."
            )
            return

    # Initialize API
    api = HfApi()

    try:
        # Create repo if it doesn't exist
        try:
            api.create_repo(repo_id=repo_id, token=token, exist_ok=True)
            print(f"Repository {repo_id} ready")
        except Exception as e:
            print(f"Note: {e}")

        # Upload file
        commit_message = api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=os.path.basename(model_path),
            repo_id=repo_id,
            token=token,
            commit_message=commit_message,
        )
        commit_hash = commit_message.oid

        print(f"Successfully uploaded to https://huggingface.co/{repo_id}")
        return commit_hash

    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")


def submit_task(
    task_id: int, hg_repo_id: str, model_filename: str, revision: str
):
    payload = json.dumps(
        {
            "task_id": task_id,
            "data": {
                "hg_repo_id": hg_repo_id,
                "model_filename": model_filename,
                "revision": revision,
            },
        }
    )
    headers = {
        "flock-api-key": FLOCK_API_KEY,
        "Content-Type": "application/json",
    }
    response = requests.request(
        "POST",
        f"{FED_LEDGER_BASE_URL}/tasks/submit-result",
        headers=headers,
        data=payload,
        timeout=30,
    )
    if response.status_code != 200:
        raise Exception(f"Failed to submit task: {response.text}")
    return response.json()



if __name__ == "__main__":
    print("=" * 70)
    print("RL TRAINING AND VALIDATION EXAMPLE")
    print("=" * 70)

    task_id = os.environ.get("TASK_ID")
    if task_id is None:
        raise Exception("TASK_ID environment variable is not set")

    # STEP 1: TRAINER - Train the model
    print("\n[Step 1] Training model...")
    log = train_mlp(
        data_dir="data",
        output_dir="runs",
        model_name="mlp_example",
        hidden=(256, 256),
        batch_size=128,
        epochs=100,
        learning_rate=1e-3,
        weight_decay=1e-5,
        validation_split=0.2,
        seed=42,
    )
    print("[Step 1] ✓ Training complete!")

    # STEP 2: TRAINER - Upload model to HuggingFace
    print("\n[Step 2] Uploading model to HuggingFace...")
    # Define model info that will be used later in validation
    model_repo_id = "your-username/mlp-example"  # Change to your repo
    model_filename = "mlp_example.onnx"

    commit_hash = upload_to_huggingface(
        model_path="runs/mlp_example.onnx",
        repo_id=model_repo_id,
        token=None,  # Will use HF_TOKEN environment variable
        commit_message="Upload trained MLP model",
    )
    print("[Step 2] ✓ Model uploaded to HuggingFace!")

    # STEP 3: VALIDATOR - Get submission to validate from backend
    print("\n[Step 3] Submit model to fed ledger")

    submit_task(
        task_id=task_id,
        hg_repo_id=model_repo_id,
        model_filename=model_filename,
        revision=commit_hash,
    )

    print("[Step 3] ✓ Model submitted to fed ledger!")