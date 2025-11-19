#!/usr/bin/env python3
"""
Line-by-Line Classification with Vision Transformer
Optimized for CPU training on SLURM clusters
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# Set matplotlib backend for non-interactive plotting
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Vision Transformer for line classification"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--samples-per-image", type=int, default=5, help="Training samples per image"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")

    # Model parameters
    parser.add_argument(
        "--embed-dim", type=int, default=256, help="Embedding dimension"
    )
    parser.add_argument(
        "--depth", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Data parameters
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed_data.npz",
        help="Path to dataset",
    )
    parser.add_argument(
        "--min-lines", type=int, default=10, help="Minimum number of lines"
    )

    # System parameters
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--num-threads", type=int, default=None, help="PyTorch threads (None=auto)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


class LineByLineDataset(Dataset):
    """
    Dataset for line-by-line classification of STM images.

    This dataset creates multiple training samples per image by randomly selecting
    different numbers of scanlines from each image. This simulates the progressive
    scanning process and helps the model learn from partial scans.

    Args:
        images: Array of shape (N, max_lines, line_width) containing STM images
        labels: Array of shape (N,) containing class labels
        min_lines: Minimum number of scanlines to include in each sample
        max_lines: Maximum number of scanlines to include in each sample
        samples_per_image: Number of different samples to generate per image
    """

    def __init__(
        self, images, labels, min_lines=5, max_lines=128, samples_per_image=10
    ):
        self.images = images.copy()  # Make a copy to avoid modifying original data
        self.labels = labels
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.samples_per_image = samples_per_image

        # Normalize each scanline independently to remove intensity variations
        self._normalize_scanlines()

    def _normalize_scanlines(self):
        """
        Normalize each scanline to zero mean and unit variance.

        This preprocessing step helps the model focus on the shape and patterns
        in the data rather than absolute intensity values, which can vary due to
        instrumental drift or tip changes during scanning.
        """
        for i in range(len(self.images)):
            for j in range(self.images.shape[1]):
                line = self.images[i, j, :]
                mean = line.mean()
                std = line.std()
                # Only normalize if std is non-zero to avoid division by zero
                if std > 1e-8:
                    self.images[i, j, :] = (line - mean) / std
                else:
                    # If std is zero, just center the data
                    self.images[i, j, :] = line - mean

    def __len__(self):
        # Total dataset size is images × samples per image
        return len(self.images) * self.samples_per_image

    def __getitem__(self, idx):
        # Determine which image this sample comes from
        img_idx = idx // self.samples_per_image

        # Randomly select how many scanlines to include in this sample
        # This creates variability and helps the model learn from partial scans
        num_lines = np.random.randint(self.min_lines, self.max_lines + 1)

        # Extract the first num_lines scanlines from the image
        partial_image = self.images[img_idx, :num_lines, :]
        label = self.labels[img_idx]

        return {
            "image": torch.FloatTensor(partial_image),
            "label": torch.LongTensor([label])[0],
            "num_lines": num_lines,  # Track length for batching
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.

    Since each sample can have a different number of scanlines (due to random
    sampling in the dataset), we need to pad shorter sequences to create uniform
    batches for the model. This function:
    1. Finds the longest sequence in the batch
    2. Creates zero-padded tensors to hold all samples
    3. Copies each sample into the padded tensor

    The transformer will process all positions, but the CLS token aggregation
    ensures that padding doesn't significantly affect the classification.

    Args:
        batch: List of dictionaries with 'image', 'label', and 'num_lines' keys

    Returns:
        Dictionary with batched 'images' and 'labels' tensors
    """
    # Find the maximum number of lines in this batch for padding
    max_lines = max(item["num_lines"] for item in batch)
    batch_size = len(batch)
    line_width = batch[0]["image"].shape[1]

    # Initialize zero-padded tensors
    images = torch.zeros(batch_size, max_lines, line_width)
    labels = torch.zeros(batch_size, dtype=torch.long)

    # Fill in the actual data (shorter sequences remain zero-padded)
    for i, item in enumerate(batch):
        num_lines = item["num_lines"]
        images[i, :num_lines] = item["image"]
        labels[i] = item["label"]

    return {"images": images, "labels": labels}


class PatchEmbedding(nn.Module):
    """
    Convert scanlines into patch embeddings for the transformer.

    This module divides each scanline into fixed-size patches and projects them
    into a higher-dimensional embedding space. This is analogous to how Vision
    Transformers (ViT) process images, but adapted for 1D scanline data.

    For example, with line_width=128 and patch_size=16:
    - Each scanline is divided into 8 patches of 16 pixels each
    - Each patch is then projected to embed_dim dimensions

    Args:
        line_width: Number of pixels in each scanline (default: 128)
        patch_size: Size of each patch in pixels (default: 16)
        embed_dim: Dimensionality of the embedding space (default: 256)
    """

    def __init__(self, line_width=128, patch_size=16, embed_dim=256):
        super().__init__()
        self.line_width = line_width
        self.patch_size = patch_size
        self.num_patches_per_line = line_width // patch_size
        self.embed_dim = embed_dim

        # Linear projection from patch_size to embed_dim
        self.projection = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        """
        Convert scanlines to patch embeddings.

        Args:
            x: Input tensor of shape (batch_size, num_lines, line_width)

        Returns:
            Patch embeddings of shape (batch_size, num_patches, embed_dim)
            where num_patches = num_lines × num_patches_per_line
        """
        batch_size, num_lines, line_width = x.shape

        # Reshape to separate patches: (batch, lines, patches_per_line, patch_size)
        x = x.reshape(batch_size, num_lines, self.num_patches_per_line, self.patch_size)

        # Flatten lines and patches into a sequence: (batch, total_patches, patch_size)
        x = x.reshape(
            batch_size, num_lines * self.num_patches_per_line, self.patch_size
        )

        # Project each patch to embedding dimension
        embeddings = self.projection(x)

        return embeddings


class PositionalEncoding(nn.Module):
    """
    Add positional information to patch embeddings using sinusoidal encoding.

    Transformers have no inherent notion of sequence order, so we add positional
    encodings to give the model information about the position of each patch.
    This is crucial for STM data where the order of scanlines matters.

    We use the sinusoidal encoding from "Attention is All You Need" (Vaswani et al.):
    - PE(pos, 2i) = sin(pos / 10000^(2i/embed_dim))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/embed_dim))

    This encoding allows the model to learn relative positions and generalizes
    well to sequence lengths not seen during training.

    Args:
        embed_dim: Dimensionality of the embeddings
        max_lines: Maximum number of scanlines expected
        patches_per_line: Number of patches per scanline
        dropout: Dropout rate applied after adding positional encoding
    """

    def __init__(self, embed_dim, max_lines=128, patches_per_line=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(p=dropout)

        # Calculate maximum sequence length (total patches)
        max_len = max_lines * patches_per_line

        # Initialize positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)

        # Create position indices: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate the division term for sinusoidal functions
        # This creates different frequencies for different dimensions
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim)
        )

        # Apply sine to even dimensions
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd dimensions
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of the model state)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input embeddings of shape (batch_size, seq_len, embed_dim)

        Returns:
            Embeddings with positional information added, same shape as input
        """
        # Add positional encoding (only up to the actual sequence length)
        x = x + self.pe[: x.size(1), :]

        # Apply dropout for regularization
        return self.dropout(x)


class VisionTransformer(nn.Module):
    """
    Vision Transformer adapted for line-by-line STM image classification.

    This model follows the ViT architecture but is adapted for sequential scanline data:
    1. Patch Embedding: Convert scanlines to patches and project to embedding space
    2. CLS Token: Add a learnable classification token at the start of sequence
    3. Positional Encoding: Add position information to embeddings
    4. Transformer Encoder: Process the sequence with self-attention
    5. Classification Head: Use the CLS token output for classification

    The model can handle variable-length inputs (different numbers of scanlines)
    thanks to the CLS token approach and positional encodings.

    Args:
        line_width: Number of pixels per scanline (default: 128)
        patch_size: Size of each patch in pixels (default: 16)
        num_classes: Number of output classes (default: 6)
        embed_dim: Embedding dimension (default: 256)
        depth: Number of transformer encoder layers (default: 6)
        num_heads: Number of attention heads (default: 8)
        mlp_ratio: Ratio of MLP hidden dim to embedding dim (default: 4)
        dropout: Dropout rate (default: 0.1)
        max_lines: Maximum number of scanlines (default: 128)
    """

    def __init__(
        self,
        line_width=128,
        patch_size=16,
        num_classes=6,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        max_lines=128,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patches_per_line = line_width // patch_size

        # Convert scanlines to patch embeddings
        self.patch_embed = PatchEmbedding(line_width, patch_size, embed_dim)

        # Learnable classification token (similar to BERT's [CLS] token)
        # This token's output will be used for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Add positional information to embeddings
        self.pos_encoding = PositionalEncoding(
            embed_dim,
            max_lines=max_lines,
            patches_per_line=self.patches_per_line,
            dropout=dropout,
        )

        # Build the transformer encoder stack
        # Using pre-normalization (norm_first=True) for better training stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,  # MLP hidden size
            dropout=dropout,
            activation="gelu",  # GELU activation (smoother than ReLU)
            batch_first=True,  # Input shape: (batch, seq, features)
            norm_first=True,  # Pre-normalization for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Final layer normalization before classification head
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head: maps CLS token embedding to class logits
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize all weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize model weights using truncated normal distribution.

        This initialization scheme helps with training stability and convergence.
        - CLS token and linear layers: truncated normal with std=0.02
        - LayerNorm: bias=0, weight=1
        """
        # Initialize CLS token
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize all linear and normalization layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Forward pass through the Vision Transformer.

        Args:
            x: Input scanlines of shape (batch_size, num_lines, line_width)

        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        batch_size = x.shape[0]

        # Step 1: Convert scanlines to patch embeddings
        # Shape: (batch_size, num_patches, embed_dim)
        x = self.patch_embed(x)

        # Step 2: Prepend CLS token to the sequence
        # CLS token will aggregate information from all patches
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat(
            [cls_tokens, x], dim=1
        )  # Shape: (batch, 1+num_patches, embed_dim)

        # Step 3: Add positional encodings
        x = self.pos_encoding(x)

        # Step 4: Process through transformer encoder layers
        # Each layer applies self-attention and feed-forward networks
        x = self.transformer(x)

        # Step 5: Extract the CLS token output (first position)
        cls_output = x[:, 0]

        # Step 6: Apply final normalization
        cls_output = self.norm(cls_output)

        # Step 7: Map to class logits
        logits = self.head(cls_output)

        return logits


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    This function performs a complete pass through the training data, computing
    gradients and updating model parameters. It tracks both loss and accuracy
    metrics for monitoring training progress.

    Args:
        model: The neural network model to train
        loader: DataLoader providing training batches
        criterion: Loss function (typically CrossEntropyLoss)
        optimizer: Optimizer for updating model parameters
        device: Device to run computations on (CPU or CUDA)

    Returns:
        Tuple of (average_loss, accuracy) for the epoch
    """
    model.train()  # Enable training mode (enables dropout, batch norm updates)
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        # Move data to the appropriate device (CPU/GPU)
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass: compute model predictions
        logits = model(images)

        # Compute loss between predictions and ground truth
        loss = criterion(logits, labels)

        # Backward pass: compute gradients
        optimizer.zero_grad()  # Clear gradients from previous step
        loss.backward()  # Backpropagate to compute gradients

        # Update model parameters using computed gradients
        optimizer.step()

        # Track metrics for monitoring
        total_loss += loss.item()
        predictions = logits.argmax(dim=1)  # Get predicted class
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Update progress bar with current metrics
        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.4f}"}
        )

    # Return average loss and accuracy for the epoch
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on validation or test data.

    This function runs the model in evaluation mode (no gradient computation)
    and computes comprehensive metrics including:
    - Average loss
    - Balanced accuracy (accounts for class imbalance)
    - AUROC (Area Under ROC Curve) for multi-class classification

    Args:
        model: The neural network model to evaluate
        loader: DataLoader providing evaluation batches
        criterion: Loss function for computing validation loss
        device: Device to run computations on (CPU or CUDA)

    Returns:
        Tuple of (average_loss, balanced_accuracy, auroc)
    """
    model.eval()  # Enable evaluation mode (disables dropout, freezes batch norm)
    total_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []

    # Disable gradient computation for efficiency and to prevent memory issues
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            # Move data to device
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass only (no backward pass in evaluation)
            logits = model(images)
            loss = criterion(logits, labels)

            # Accumulate loss
            total_loss += loss.item()

            # Convert logits to probabilities and predictions
            probs = F.softmax(logits, dim=1)  # Convert to probabilities
            preds = logits.argmax(dim=1)  # Get predicted class

            # Collect all predictions and labels for metric computation
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert lists to numpy arrays for sklearn metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Compute evaluation metrics
    # Balanced accuracy: accounts for class imbalance by averaging per-class recall
    accuracy = balanced_accuracy_score(all_labels, all_preds)

    # AUROC: measures the model's ability to distinguish between classes
    # 'ovr' (one-vs-rest) strategy for multi-class, 'macro' averaging for equal weight per class
    auroc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")

    return total_loss / len(loader), accuracy, auroc


def plot_training_history(history, output_dir):
    """Plot training history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Balanced Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["val_auroc"], label="Validation", color="green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUROC")
    axes[2].set_title("Validation AUROC")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """
    Main training function that orchestrates the entire training pipeline.

    This function handles:
    1. Setup: logging, random seeds, output directories
    2. Data loading and splitting (train/val/test)
    3. Model creation and initialization
    4. Training loop with validation
    5. Checkpointing and result saving
    """
    args = parse_args()

    # ============================================================================
    # SETUP: Random seeds, directories, and logging
    # ============================================================================

    # Set random seeds for reproducibility across runs
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory for saving models, logs, and plots
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Setup logging to both console and file
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def log(msg):
        """
        Log message to both console and file with timestamp.

        Args:
            msg: Message string to log
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {msg}"
        print(log_msg)
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")

    log("=" * 80)
    log("Starting Training")
    log("=" * 80)
    log(f"Arguments: {vars(args)}")

    # ============================================================================
    # DEVICE AND PERFORMANCE CONFIGURATION
    # ============================================================================

    # CPU optimization: control number of threads for parallel operations
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        log(f"Set PyTorch threads to: {args.num_threads}")
    else:
        log(f"Using default PyTorch threads: {torch.get_num_threads()}")

    # Device configuration: use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    # ============================================================================
    # DATA LOADING AND SPLITTING
    # ============================================================================

    # Load the preprocessed dataset (NPZ format containing images and labels)
    log(f"Loading dataset from {args.data_path}...")
    data = np.load(args.data_path)
    all_images = data[data.files[0]]  # STM images as numpy array
    all_labels = data[data.files[1]]  # Corresponding class labels

    log(f"Total dataset: {all_images.shape[0]} images")
    log(f"Image size: {all_images.shape[1]} x {all_images.shape[2]}")
    log(f"Number of classes: {len(np.unique(all_labels))}")

    # Split dataset into train/val/test sets
    # First split: separate 15% for test set
    train_val_images, test_images, train_val_labels, test_labels = train_test_split(
        all_images,
        all_labels,
        test_size=0.15,
        random_state=args.seed,
        stratify=all_labels,
    )

    # Second split: from remaining 85%, take ~15% for validation (0.176 * 0.85 ≈ 0.15)
    # Final split: 70% train, 15% val, 15% test
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_val_images,
        train_val_labels,
        test_size=0.176,
        random_state=args.seed,
        stratify=train_val_labels,
    )

    log(f"Train set: {train_images.shape[0]} images")
    log(f"Val set: {val_images.shape[0]} images")
    log(f"Test set: {test_images.shape[0]} images")

    # ============================================================================
    # DATASET AND DATALOADER CREATION
    # ============================================================================

    # Create PyTorch datasets that handle line-by-line sampling and normalization
    train_dataset = LineByLineDataset(
        train_images,
        train_labels,
        min_lines=args.min_lines,
        samples_per_image=args.samples_per_image,
    )
    val_dataset = LineByLineDataset(
        val_images,
        val_labels,
        min_lines=args.min_lines,
        samples_per_image=args.samples_per_image,
    )

    # Create dataloaders for batching and parallel data loading
    # Note: shuffle=True for training (randomization), False for validation (consistency)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,  # Custom collate for variable-length sequences
        num_workers=args.num_workers,  # Parallel data loading
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    log(f"Training batches: {len(train_loader)}")
    log(f"Validation batches: {len(val_loader)}")

    # ============================================================================
    # MODEL CREATION AND INITIALIZATION
    # ============================================================================

    # Create Vision Transformer model with specified architecture parameters
    num_classes = len(np.unique(all_labels))
    model = VisionTransformer(
        line_width=128,  # Fixed scanline width from preprocessing
        patch_size=16,  # Each patch contains 16 pixels
        num_classes=num_classes,
        embed_dim=args.embed_dim,  # Embedding dimension
        depth=args.depth,  # Number of transformer layers
        num_heads=args.num_heads,  # Number of attention heads per layer
        mlp_ratio=4,  # MLP hidden size = 4 × embed_dim
        dropout=args.dropout,  # Dropout rate for regularization
        max_lines=128,  # Maximum sequence length
    ).to(device)

    # Count and log model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model created with {num_params:,} trainable parameters")

    # ============================================================================
    # TRAINING SETUP: Loss, Optimizer, Scheduler
    # ============================================================================

    # Loss function: Cross-entropy for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Optimizer: AdamW (Adam with decoupled weight decay)
    # AdamW is better than Adam for transformers as it properly regularizes
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,  # L2 regularization
    )

    # Learning rate scheduler: Cosine annealing
    # Gradually reduces learning rate following a cosine curve
    # This helps with final convergence and prevents overshooting
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ============================================================================
    # CHECKPOINT LOADING (Resume Training)
    # ============================================================================

    # Initialize training state variables
    start_epoch = 0
    best_val_auroc = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auroc": [],
    }

    # Check if we should resume from a previous checkpoint
    checkpoint_path = output_dir / "best_model.pt"
    if args.resume and checkpoint_path.exists():
        log(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Restore model and optimizer state
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore training progress
        start_epoch = checkpoint["epoch"] + 1
        best_val_auroc = checkpoint["val_auroc"]
        if "history" in checkpoint:
            history = checkpoint["history"]

        log(f"Resumed from epoch {start_epoch}, best AUROC: {best_val_auroc:.4f}")

    # ============================================================================
    # TRAINING LOOP
    # ============================================================================

    log("=" * 80)
    log("Starting Training Loop")
    log("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        log(f"\nEpoch {epoch + 1}/{args.epochs}")
        epoch_start = datetime.now()

        # ---- Training Phase ----
        # Run one epoch of training (forward, backward, update)
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # ---- Validation Phase ----
        # Evaluate on validation set (no gradient computation)
        val_loss, val_acc, val_auroc = evaluate(model, val_loader, criterion, device)

        # ---- Learning Rate Schedule Update ----
        # Update learning rate according to cosine schedule
        scheduler.step()

        # ---- Record Metrics ----
        # Keep track of all metrics for plotting
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auroc"].append(val_auroc)

        epoch_time = (datetime.now() - epoch_start).total_seconds()

        # ---- Log Progress ----
        log(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        log(
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUROC: {val_auroc:.4f}"
        )
        log(f"Epoch time: {epoch_time:.1f}s")

        # Estimate time remaining (helpful for long training runs)
        if epoch > start_epoch:
            avg_epoch_time = epoch_time
            remaining_epochs = args.epochs - (epoch + 1)
            eta_seconds = remaining_epochs * avg_epoch_time
            eta_hours = eta_seconds / 3600
            log(f"Estimated time remaining: {eta_hours:.1f} hours")

        # ---- Model Checkpointing ----
        # Save model if it achieves the best validation AUROC so far
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auroc": val_auroc,
                    "history": history,
                },
                checkpoint_path,
            )
            log(f"Saved best model with AUROC: {val_auroc:.4f}")

        # ---- Periodic Plotting ----
        # Save training curves every 5 epochs to monitor progress
        if (epoch + 1) % 5 == 0:
            plot_training_history(history, output_dir)

    # ============================================================================
    # FINAL EVALUATION AND SAVING
    # ============================================================================

    log("=" * 80)
    log("Training Complete!")
    log("=" * 80)
    log(f"Best validation AUROC: {best_val_auroc:.4f}")

    # Plot final training history curves
    plot_training_history(history, output_dir)

    # ---- Save Model Configuration ----
    # Store architecture hyperparameters for model reconstruction
    model_config = {
        "line_width": 128,
        "patch_size": 16,
        "num_classes": num_classes,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "mlp_ratio": 4,
        "dropout": args.dropout,
        "max_lines": 128,
    }

    # Save config as JSON for easy inspection
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    # ---- Save Final Model ----
    # Save the model from the last epoch (not necessarily the best)
    # This can be useful for analyzing training dynamics or resuming
    torch.save(
        {
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "best_val_auroc": best_val_auroc,
            "history": history,
        },
        output_dir / "final_model.pt",
    )

    log(f"All outputs saved to: {output_dir}")
    log("Done!")


if __name__ == "__main__":
    main()
