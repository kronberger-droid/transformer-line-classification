# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Line-by-Line Classification with Vision Transformer
#
# A clean implementation of sequential scanline classification using a Vision Transformer (ViT) architecture.
#
# ## Approach
#
# - **Input**: Partial scan with j scanlines (where j = 1, 2, ..., max_lines)
# - **Architecture**: Vision Transformer that treats scanline patches as tokens
# - **Output**: Classification prediction based on lines seen so far
#
# ## Key Components
#
# 1. Patch embedding: Convert scanlines to tokens
# 2. Positional encoding: Encode spatial + temporal position
# 3. Transformer encoder: Self-attention across all patches
# 4. Classification head: Predict class from aggregated features

# %% [markdown]
# ## 1. Setup

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
import seaborn as sns
from tqdm.auto import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# %% [markdown]
# ## 2. Dataset
#
# The dataset should contain images where each image has:
# - Shape: [num_lines, line_width] (e.g., [128, 128])
# - Label: Integer class label

# %%
class LineByLineDataset(Dataset):
    """
    Dataset for line-by-line classification.
    
    For each image, we create multiple samples by varying the number of visible lines.
    This simulates the real-time scanning scenario.
    """
    
    def __init__(self, images, labels, min_lines=5, max_lines=128, samples_per_image=10):
        """
        Args:
            images: Array of shape [N, max_lines, line_width]
            labels: Array of shape [N] with class labels
            min_lines: Minimum number of lines to use
            max_lines: Maximum number of lines (full image)
            samples_per_image: How many random cutoffs to generate per image
        """
        self.images = images
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
        # Determine which image and which random cutoff
        img_idx = idx // self.samples_per_image
        
        # Random number of lines between min and max
        num_lines = np.random.randint(self.min_lines, self.max_lines + 1)
        
        # Get partial image
        partial_image = self.images[img_idx, :num_lines, :]
        label = self.labels[img_idx]
        
        return {
            'image': torch.FloatTensor(partial_image),
            'label': torch.LongTensor([label])[0],
            'num_lines': num_lines
        }


def read_dataset():
    """
    Read dataset for testing
    """
    data = np.load('data/processed_data.npz')

    images = data[data.files[0]]
    labels = data[data.files[1]]
    return images, labels


# %%
# Load the real dataset
print("Loading dataset from processed_data.npz...")
data = np.load('data/processed_data.npz')
all_images = data[data.files[0]]
all_labels = data[data.files[1]]

print(f"Total dataset: {all_images.shape[0]} images")
print(f"Image size: {all_images.shape[1]} x {all_images.shape[2]}")
print(f"Number of classes: {len(np.unique(all_labels))}")
print(f"Class distribution:")
for cls in np.unique(all_labels):
    count = np.sum(all_labels == cls)
    print(f"  Class {cls}: {count} samples ({100*count/len(all_labels):.2f}%)")

# Split into train/val/test (70/15/15)
from sklearn.model_selection import train_test_split

# First split: separate test set
train_val_images, test_images, train_val_labels, test_labels = train_test_split(
    all_images, all_labels, test_size=0.15, random_state=42, stratify=all_labels
)

# Second split: separate validation set
train_images, val_images, train_labels, val_labels = train_test_split(
    train_val_images, train_val_labels, test_size=0.176, random_state=42, stratify=train_val_labels
)  # 0.176 * 0.85 ≈ 0.15 of total

print(f"\nTrain set: {train_images.shape[0]} images")
print(f"Val set: {val_images.shape[0]} images")
print(f"Test set: {test_images.shape[0]} images")


# %% [markdown]
# ## 3. Model Architecture
#
# ### 3.1 Patch Embedding

# %%
class PatchEmbedding(nn.Module):
    """
    Convert scanlines into patch embeddings.
    
    Each scanline is divided into patches, and each patch is linearly projected
    to the embedding dimension.
    """
    
    def __init__(self, line_width=128, patch_size=16, embed_dim=256):
        super().__init__()
        self.line_width = line_width
        self.patch_size = patch_size
        self.num_patches_per_line = line_width // patch_size
        self.embed_dim = embed_dim
        
        # Linear projection for patches
        self.projection = nn.Linear(patch_size, embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_lines, line_width]
        
        Returns:
            embeddings: [batch_size, num_lines * num_patches_per_line, embed_dim]
        """
        batch_size, num_lines, line_width = x.shape
        
        # Reshape into patches: [batch, num_lines, num_patches_per_line, patch_size]
        x = x.reshape(batch_size, num_lines, self.num_patches_per_line, self.patch_size)
        
        # Flatten line and patch dimensions: [batch, num_lines * num_patches_per_line, patch_size]
        x = x.reshape(batch_size, num_lines * self.num_patches_per_line, self.patch_size)
        
        # Project to embedding dimension: [batch, num_tokens, embed_dim]
        embeddings = self.projection(x)
        
        return embeddings


# %% [markdown]
# ### 3.2 Positional Encoding

# %%
class PositionalEncoding(nn.Module):
    """
    Add positional information to patch embeddings.
    
    We need to encode two types of position:
    1. Which line (temporal: 0, 1, 2, ... j-1)
    2. Which patch within the line (spatial: 0, 1, 2, ... num_patches-1)
    """
    
    def __init__(self, embed_dim, max_lines=128, patches_per_line=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_lines = max_lines
        self.patches_per_line = patches_per_line
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        max_len = max_lines * patches_per_line
        pe = torch.zeros(max_len, embed_dim)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but should be moved to GPU)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_tokens, embed_dim]
        
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


# %% [markdown]
# ### 3.3 Vision Transformer

# %%
class VisionTransformer(nn.Module):
    """
    Vision Transformer for line-by-line classification.
    
    Architecture:
    1. Patch embedding
    2. Add CLS token
    3. Positional encoding
    4. Transformer encoder layers
    5. Classification head (using CLS token)
    """
    
    def __init__(
        self,
        line_width=128,
        patch_size=16,
        num_classes=4,
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
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(line_width, patch_size, embed_dim)
        
        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            embed_dim, 
            max_lines=max_lines, 
            patches_per_line=self.patches_per_line,
            dropout=dropout
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following ViT paper."""
        # Initialize CLS token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize linear layers
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
        Args:
            x: [batch_size, num_lines, line_width]
        
        Returns:
            logits: [batch_size, num_classes]
        """
        batch_size = x.shape[0]
        
        # Patch embedding: [batch, num_lines, line_width] -> [batch, num_tokens, embed_dim]
        x = self.patch_embed(x)
        
        # Prepend CLS token: [batch, num_tokens + 1, embed_dim]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Take CLS token output
        cls_output = x[:, 0]
        
        # Normalize and classify
        cls_output = self.norm(cls_output)
        logits = self.head(cls_output)
        
        return logits


# %%
# Create model
model = VisionTransformer(
    line_width=128,
    patch_size=16,
    num_classes=6,  # Updated to 6 classes based on the dataset
    embed_dim=256,
    depth=6,
    num_heads=8,
    mlp_ratio=4,
    dropout=0.1,
    max_lines=128
).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model created with {num_params:,} trainable parameters")

# Test forward pass
dummy_input = torch.randn(2, 50, 128).to(device)
output = model(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")


# %% [markdown]
# ## 4. Training Setup

# %%
def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads to the maximum length in the batch.
    """
    max_lines = max(item['num_lines'] for item in batch)
    batch_size = len(batch)
    line_width = batch[0]['image'].shape[1]
    
    # Create padded tensors
    images = torch.zeros(batch_size, max_lines, line_width)
    labels = torch.zeros(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        num_lines = item['num_lines']
        images[i, :num_lines] = item['image']
        labels[i] = item['label']
    
    return {'images': images, 'labels': labels}


# Create datasets
# For real training: samples_per_image=5
train_dataset = LineByLineDataset(train_images, train_labels, min_lines=10, samples_per_image=2)
val_dataset = LineByLineDataset(val_images, val_labels, min_lines=10, samples_per_image=2)
test_dataset = LineByLineDataset(test_images, test_labels, min_lines=10, samples_per_image=2)

# Create dataloaders
# For real training: batch_size=32
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# %%
# Training configuration
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

# Learning rate scheduler
# For real training: num_epochs=50
num_epochs = 2
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


# %% [markdown]
# ## 5. Training Loop

# %%
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
        
        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})
    
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
    
    # Calculate metrics
    accuracy = balanced_accuracy_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    
    return total_loss / len(loader), accuracy, auroc


# %%
# Training loop
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'val_auroc': []
}

best_val_auroc = 0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    
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
    
    # Print summary
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUROC: {val_auroc:.4f}")
    
    # Save best model
    if val_auroc > best_val_auroc:
        best_val_auroc = val_auroc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auroc': val_auroc,
        }, 'best_model.pt')
        print(f"Saved best model with AUROC: {val_auroc:.4f}")

# %% [markdown]
# ## 6. Visualization

# %%
# Plot training history
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Loss
axes[0].plot(history['train_loss'], label='Train')
axes[0].plot(history['val_loss'], label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(history['train_acc'], label='Train')
axes[1].plot(history['val_acc'], label='Validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Balanced Accuracy')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# AUROC
axes[2].plot(history['val_auroc'], label='Validation', color='green')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('AUROC')
axes[2].set_title('Validation AUROC')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()


# %% [markdown]
# ## 7. Evaluation: Performance vs. Number of Lines
#
# Key analysis: How does performance improve as we see more scanlines?

# %%
def evaluate_vs_num_lines(model, images, labels, device, line_steps=None):
    """
    Evaluate model performance as a function of number of scanlines.
    
    Args:
        model: Trained model
        images: Full images [N, max_lines, line_width]
        labels: Labels [N]
        device: Device to run on
        line_steps: List of line counts to evaluate at (default: [10, 20, 30, ..., 128])
    
    Returns:
        Dictionary with performance metrics at each line count
    """
    if line_steps is None:
        line_steps = list(range(10, 129, 10))  # Every 10 lines
    
    model.eval()
    results = {'num_lines': [], 'accuracy': [], 'auroc': []}
    
    for num_lines in tqdm(line_steps, desc='Evaluating vs. num lines'):
        all_labels = []
        all_probs = []
        all_preds = []
        
        with torch.no_grad():
            for i in range(0, len(images), 32):  # Batch size 32
                batch_images = images[i:i+32, :num_lines, :]
                batch_labels = labels[i:i+32]
                
                batch_images = torch.FloatTensor(batch_images).to(device)
                
                logits = model(batch_images)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                
                all_labels.extend(batch_labels)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        
        accuracy = balanced_accuracy_score(all_labels, all_preds)
        auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        
        results['num_lines'].append(num_lines)
        results['accuracy'].append(accuracy)
        results['auroc'].append(auroc)
    
    return results


# %%
# Load best model
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate on test set
results = evaluate_vs_num_lines(model, test_images, test_labels, device)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# AUROC vs. number of lines
ax1.plot(results['num_lines'], results['auroc'], marker='o', linewidth=2, markersize=6)
ax1.set_xlabel('Number of Scanlines')
ax1.set_ylabel('AUROC')
ax1.set_title('AUROC vs. Number of Scanlines')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

# Accuracy vs. number of lines
ax2.plot(results['num_lines'], results['accuracy'], marker='o', linewidth=2, markersize=6, color='green')
ax2.set_xlabel('Number of Scanlines')
ax2.set_ylabel('Balanced Accuracy')
ax2.set_title('Accuracy vs. Number of Scanlines')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('performance_vs_lines.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nPerformance Summary:")
for i, num_lines in enumerate(results['num_lines']):
    print(f"Lines: {num_lines:3d} | AUROC: {results['auroc'][i]:.4f} | Accuracy: {results['accuracy'][i]:.4f}")

# %% [markdown]
# ## 8. Confusion Matrix

# %%
# Get predictions on full test set
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        images = batch['images'].to(device)
        labels = batch['labels']
        
        logits = model(images)
        preds = logits.argmax(dim=1)
        
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[f'Class {i}' for i in range(6)],
            yticklabels=[f'Class {i}' for i in range(6)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 9. Attention Visualization (Optional)
#
# TODO: Extract and visualize attention maps to understand what the model focuses on.

# %%
# TODO: Modify model to return attention weights
# TODO: Visualize which patches/lines the model attends to
# This helps interpret model decisions

# %% [markdown]
# ## 10. Model Export
#
# Export model for deployment or further analysis.

# %%
# Save model architecture and weights
import json

model_config = {
    'line_width': 128,
    'patch_size': 16,
    'num_classes': 6,  # Updated to 6 classes
    'embed_dim': 256,
    'depth': 6,
    'num_heads': 8,
    'mlp_ratio': 4,
    'dropout': 0.1,
    'max_lines': 128
}

with open('model_config.json', 'w') as f:
    json.dump(model_config, f, indent=2)

# Save final model
torch.save({
    'model_config': model_config,
    'model_state_dict': model.state_dict(),
    'best_val_auroc': best_val_auroc
}, 'final_model.pt')

print("Model saved successfully!")
print(f"Best validation AUROC: {best_val_auroc:.4f}")

# %% [markdown]
# ## Summary
#
# This notebook implements a Vision Transformer for line-by-line classification. The key features:
#
# 1. **Patch-based tokenization**: Each scanline is divided into patches
# 2. **Positional encoding**: Encodes both spatial (within-line) and temporal (which line) information
# 3. **Transformer encoder**: Self-attention across all patches from all visible lines
# 4. **CLS token**: Global representation for classification
#
# ### Next Steps:
#
# 1. Replace dummy data with actual dataset
# 2. Tune hyperparameters (patch size, model depth, etc.)
# 3. Implement attention visualization
# 4. Compare with baseline methods
# 5. Port to Rust/Burn if needed for deployment
