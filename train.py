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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description='Train Vision Transformer for line classification')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--samples-per-image', type=int, default=5, help='Training samples per image')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay')

    # Model parameters
    parser.add_argument('--embed-dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Data parameters
    parser.add_argument('--data-path', type=str, default='data/processed_data.npz', help='Path to dataset')
    parser.add_argument('--min-lines', type=int, default=10, help='Minimum number of lines')

    # System parameters
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--num-threads', type=int, default=None, help='PyTorch threads (None=auto)')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser.parse_args()


class LineByLineDataset(Dataset):
    """Dataset for line-by-line classification."""

    def __init__(self, images, labels, min_lines=5, max_lines=128, samples_per_image=10):
        self.images = images.copy()  # Make a copy to avoid modifying original
        self.labels = labels
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.samples_per_image = samples_per_image

        # Normalize each scanline independently
        self._normalize_scanlines()

    def _normalize_scanlines(self):
        """Normalize each scanline to zero mean and unit variance."""
        for i in range(len(self.images)):
            for j in range(self.images.shape[1]):
                line = self.images[i, j, :]
                mean = line.mean()
                std = line.std()
                if std > 1e-8:
                    self.images[i, j, :] = (line - mean) / std
                else:
                    self.images[i, j, :] = line - mean

    def __len__(self):
        return len(self.images) * self.samples_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.samples_per_image
        num_lines = np.random.randint(self.min_lines, self.max_lines + 1)
        partial_image = self.images[img_idx, :num_lines, :]
        label = self.labels[img_idx]

        return {
            'image': torch.FloatTensor(partial_image),
            'label': torch.LongTensor([label])[0],
            'num_lines': num_lines
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    max_lines = max(item['num_lines'] for item in batch)
    batch_size = len(batch)
    line_width = batch[0]['image'].shape[1]

    images = torch.zeros(batch_size, max_lines, line_width)
    labels = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        num_lines = item['num_lines']
        images[i, :num_lines] = item['image']
        labels[i] = item['label']

    return {'images': images, 'labels': labels}


class PatchEmbedding(nn.Module):
    """Convert scanlines into patch embeddings."""

    def __init__(self, line_width=128, patch_size=16, embed_dim=256):
        super().__init__()
        self.line_width = line_width
        self.patch_size = patch_size
        self.num_patches_per_line = line_width // patch_size
        self.embed_dim = embed_dim
        self.projection = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        batch_size, num_lines, line_width = x.shape
        x = x.reshape(batch_size, num_lines, self.num_patches_per_line, self.patch_size)
        x = x.reshape(batch_size, num_lines * self.num_patches_per_line, self.patch_size)
        embeddings = self.projection(x)
        return embeddings


class PositionalEncoding(nn.Module):
    """Add positional information to patch embeddings."""

    def __init__(self, embed_dim, max_lines=128, patches_per_line=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(p=dropout)

        max_len = max_lines * patches_per_line
        pe = torch.zeros(max_len, embed_dim)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class VisionTransformer(nn.Module):
    """Vision Transformer for line-by-line classification."""

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
        max_lines=128
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patches_per_line = line_width // patch_size

        self.patch_embed = PatchEmbedding(line_width, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_encoding = PositionalEncoding(
            embed_dim, max_lines=max_lines,
            patches_per_line=self.patches_per_line, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        cls_output = x[:, 0]
        cls_output = self.norm(cls_output)
        logits = self.head(cls_output)
        return logits


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    accuracy = balanced_accuracy_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

    return total_loss / len(loader), accuracy, auroc


def plot_training_history(history, output_dir):
    """Plot training history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Balanced Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history['val_auroc'], label='Validation', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUROC')
    axes[2].set_title('Validation AUROC')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Setup logging
    log_file = output_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    def log(msg):
        """Log to console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {msg}"
        print(log_msg)
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')

    log("=" * 80)
    log("Starting Training")
    log("=" * 80)
    log(f"Arguments: {vars(args)}")

    # CPU optimization
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        log(f"Set PyTorch threads to: {args.num_threads}")
    else:
        log(f"Using default PyTorch threads: {torch.get_num_threads()}")

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Using device: {device}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load dataset
    log(f"Loading dataset from {args.data_path}...")
    data = np.load(args.data_path)
    all_images = data[data.files[0]]
    all_labels = data[data.files[1]]

    log(f"Total dataset: {all_images.shape[0]} images")
    log(f"Image size: {all_images.shape[1]} x {all_images.shape[2]}")
    log(f"Number of classes: {len(np.unique(all_labels))}")

    # Split dataset
    train_val_images, test_images, train_val_labels, test_labels = train_test_split(
        all_images, all_labels, test_size=0.15, random_state=args.seed, stratify=all_labels
    )
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_val_images, train_val_labels, test_size=0.176, random_state=args.seed, stratify=train_val_labels
    )

    log(f"Train set: {train_images.shape[0]} images")
    log(f"Val set: {val_images.shape[0]} images")
    log(f"Test set: {test_images.shape[0]} images")

    # Create datasets
    train_dataset = LineByLineDataset(
        train_images, train_labels,
        min_lines=args.min_lines, samples_per_image=args.samples_per_image
    )
    val_dataset = LineByLineDataset(
        val_images, val_labels,
        min_lines=args.min_lines, samples_per_image=args.samples_per_image
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers
    )

    log(f"Training batches: {len(train_loader)}")
    log(f"Validation batches: {len(val_loader)}")

    # Create model
    num_classes = len(np.unique(all_labels))
    model = VisionTransformer(
        line_width=128, patch_size=16, num_classes=num_classes,
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
        mlp_ratio=4, dropout=args.dropout, max_lines=128
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model created with {num_params:,} trainable parameters")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume from checkpoint if exists
    start_epoch = 0
    best_val_auroc = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auroc': []
    }

    checkpoint_path = output_dir / 'best_model.pt'
    if args.resume and checkpoint_path.exists():
        log(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_auroc = checkpoint['val_auroc']
        if 'history' in checkpoint:
            history = checkpoint['history']
        log(f"Resumed from epoch {start_epoch}, best AUROC: {best_val_auroc:.4f}")

    # Training loop
    log("=" * 80)
    log("Starting Training Loop")
    log("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        log(f"\nEpoch {epoch + 1}/{args.epochs}")
        epoch_start = datetime.now()

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, val_auroc = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auroc'].append(val_auroc)

        epoch_time = (datetime.now() - epoch_start).total_seconds()

        # Log epoch summary
        log(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        log(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUROC: {val_auroc:.4f}")
        log(f"Epoch time: {epoch_time:.1f}s")

        # Estimate remaining time
        if epoch > start_epoch:
            avg_epoch_time = epoch_time
            remaining_epochs = args.epochs - (epoch + 1)
            eta_seconds = remaining_epochs * avg_epoch_time
            eta_hours = eta_seconds / 3600
            log(f"Estimated time remaining: {eta_hours:.1f} hours")

        # Save best model
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auroc': val_auroc,
                'history': history,
            }, checkpoint_path)
            log(f"Saved best model with AUROC: {val_auroc:.4f}")

        # Save training history plot every 5 epochs
        if (epoch + 1) % 5 == 0:
            plot_training_history(history, output_dir)

    # Final evaluation and plots
    log("=" * 80)
    log("Training Complete!")
    log("=" * 80)
    log(f"Best validation AUROC: {best_val_auroc:.4f}")

    # Plot final training history
    plot_training_history(history, output_dir)

    # Save model config
    model_config = {
        'line_width': 128, 'patch_size': 16, 'num_classes': num_classes,
        'embed_dim': args.embed_dim, 'depth': args.depth,
        'num_heads': args.num_heads, 'mlp_ratio': 4,
        'dropout': args.dropout, 'max_lines': 128
    }

    with open(output_dir / 'model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)

    # Save final model
    torch.save({
        'model_config': model_config,
        'model_state_dict': model.state_dict(),
        'best_val_auroc': best_val_auroc,
        'history': history,
    }, output_dir / 'final_model.pt')

    log(f"All outputs saved to: {output_dir}")
    log("Done!")


if __name__ == '__main__':
    main()
