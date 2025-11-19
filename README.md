# Line-by-Line Classification with Vision Transformer

A Vision Transformer implementation for classifying STM (Scanning Tunneling Microscope) images based on sequential scanline data.

## Dataset

- **Total images**: 18,932
- **Image size**: 128 × 128 (128 scanlines, 128 pixels per line)
- **Number of classes**: 6
- **Classes**: 0, 1, 2, 3, 4, 5

### Class Distribution:
- Class 0: 1,664 images (8.79%)
- Class 1: 1,482 images (7.83%)
- Class 2: 6,830 images (36.08%)
- Class 3: 3,202 images (16.91%)
- Class 4: 1,332 images (7.04%)
- Class 5: 4,422 images (23.36%)

### Vision Transformer (ViT)

The model processes scanlines sequentially:

1. **Patch Embedding**: Each scanline is divided into 8 patches (16 pixels each)
2. **CLS Token**: A learnable token prepended for classification
3. **Positional Encoding**: Sinusoidal encoding for patch positions
4. **Transformer Encoder**: 6 layers of multi-head self-attention (8 heads)
5. **Classification Head**: Linear layer projecting CLS token to 6 classes

### Model Parameters:
- **Embedding dimension**: 256
- **Number of layers**: 6
- **Attention heads**: 8
- **Patch size**: 16 pixels
- **Total parameters**: ~2.7 million

## Key Features

### Line-by-Line Processing

The model can make predictions with any number of scanlines (5-128):
- Simulates real-time scanning scenario
- Each training sample uses a random number of lines
- Evaluation measures performance at different scan completeness levels

### Training Configuration

- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.05)
- **Scheduler**: Cosine annealing
- **Loss**: Cross-entropy
- **Batch size**: 32
- **Epochs**: 50
- **Data augmentation**: Random scanline cutoffs

Check the exploration notebooks (`explore_dataset.ipynb`) for more details about the data structure and processing.
