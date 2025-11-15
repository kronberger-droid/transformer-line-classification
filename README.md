# Line-by-Line Classification with Vision Transformer

A Vision Transformer implementation for classifying STM (Scanning Tunneling Microscope) images based on sequential scanline data.

## Dataset

Your dataset has been successfully processed from the `.mat` file:

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

## Files

1. **`vit_line_classification.ipynb`** - Main training notebook (ready to run!)
2. **`explore_dataset.ipynb`** - Dataset exploration notebook
3. **`processed_data.npz`** - Processed dataset (images + labels)
4. **`sample_images.png`** - Visualization of sample STM images
5. **`understand_structure.py`** - Script that processed the .mat file
6. **`quick_explore.py`** - Quick data exploration script

## Getting Started

### 1. Start Jupyter Lab

```bash
jupyter lab
```

### 2. Open the Main Notebook

Open `vit_line_classification.ipynb` and run the cells sequentially.

The notebook is now configured to:
- Load your real data from `processed_data.npz`
- Split into train (70%), validation (15%), and test (15%)
- Train a Vision Transformer with 6 output classes
- Evaluate performance vs. number of scanlines

## Model Architecture

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

## Expected Outputs

After training, you'll get:

1. **Training history plots**: Loss, accuracy, and AUROC over epochs
2. **Performance vs. scanlines**: How accuracy improves with more lines
3. **Confusion matrix**: Classification performance per class
4. **Saved models**: Best model checkpoint + final model

## Understanding the Data

The STM images show different surface features:
- Different classes likely represent different tip states or surface conditions
- Class 2 is most common (36% of data)
- The imbalanced distribution is handled by using balanced accuracy metric

## Next Steps

1. **Run the notebook** to train your first model
2. **Tune hyperparameters**:
   - Try different patch sizes (8, 16, 32)
   - Adjust model depth (4, 8, 12 layers)
   - Modify embedding dimension (128, 256, 512)
3. **Analyze results**:
   - Check which classes are confused
   - Visualize attention patterns
   - Evaluate early prediction capability
4. **Experiment with training**:
   - Try different data augmentation
   - Adjust learning rate and batch size
   - Use class weights to handle imbalance

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size (try 16 or 8)
- Reduce model size (embed_dim=128, depth=4)
- Use gradient checkpointing

### Poor Performance
- Check if data normalization is working correctly
- Verify class labels are correct
- Increase training epochs
- Try different learning rates (1e-3, 1e-5)

### Slow Training
- Ensure GPU is being used (check "Using device: cuda")
- Reduce `samples_per_image` in dataset (currently 5)
- Use smaller model for faster iteration

## Data Format

The processed data is stored as:

```python
data = np.load('processed_data.npz')
images = data['images']  # shape: (18932, 128, 128)
labels = data['labels']  # shape: (18932,)
```

Each image is a 2D array where:
- **Rows (dim 1)**: Scanline number (0-127, top to bottom)
- **Columns (dim 2)**: Pixel position in scanline (0-127, left to right)
- **Values**: Normalized pixel intensities

## Citation

If this is based on or related to:
> Gordon et al., "Embedding Human Heuristics in Machine-Learning-Enabled Probe Microscopy"

Please cite the original paper appropriately.

## Questions?

Check the exploration notebooks (`explore_dataset.ipynb`) for more details about the data structure and processing.
