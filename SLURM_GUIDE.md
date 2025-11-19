# SLURM Training Guide

Quick guide for running Vision Transformer training on SLURM clusters with CPU.

## Files Created

1. **`train.py`** - Clean training script with CPU optimizations and CLI arguments
2. **`train_test.slurm`** - Test job (2 epochs, small batch)
3. **`train_full.slurm`** - Full training job (50 epochs, production settings)

## Before First Run

### 1. Check Your Cluster Configuration

```bash
# Check available partitions
sinfo

# Check partition details (replace 'cpu' with your partition name)
scontrol show partition cpu

# Check your current limits
squeue -u $USER
```

### 2. Update SLURM Scripts

Edit both `train_test.slurm` and `train_full.slurm`:

```bash
# Line 9: Update partition name
#SBATCH --partition=YOUR_PARTITION_NAME

# Line 7: Adjust CPU cores as needed
#SBATCH --cpus-per-task=64  # or 32, 96, etc.

# Lines 32-34: Uncomment and adjust module loading
module load python/3.10  # or your cluster's Python module

# Lines 38-41: Uncomment and adjust environment activation
source /path/to/your/venv/bin/activate
# OR
conda activate your_env_name
```

### 3. Test Python Environment

```bash
# Make sure PyTorch and dependencies are installed
python -c "import torch; print(torch.__version__)"
python -c "import sklearn; print('sklearn OK')"
python -c "import tqdm; print('tqdm OK')"
```

## Running Jobs

### Step 1: Test Run (2 epochs)

**Why?** Verify everything works before committing to a long job.

```bash
# Submit test job
sbatch train_test.slurm

# Monitor job
squeue -u $USER

# Watch output in real-time
tail -f logs/test_<job_id>.out

# Check for errors
tail -f logs/test_<job_id>.err
```

**What to check after test:**
- ✓ Job completed without errors
- ✓ CPU threads are properly set (check log output)
- ✓ Checkpoint created: `outputs/test_run/best_model.pt`
- ✓ Plots created: `outputs/test_run/training_history.png`
- ✓ Check time per epoch (multiply by 25 for 50-epoch estimate)

### Step 2: Full Training (50 epochs)

Once test passes:

```bash
# Submit full training job
sbatch train_full.slurm

# Monitor progress
tail -f logs/full_<job_id>.out
```

## Resume Training

Both scripts support resuming from checkpoints:

```bash
# If job was interrupted, just resubmit
sbatch train_full.slurm  # Will automatically resume from best_model.pt

# Or resume test from where it left off
sbatch train_test.slurm
```

The `--resume` flag is already enabled in both scripts. Training will:
1. Load model weights from `best_model.pt`
2. Restore optimizer state
3. Continue from the next epoch
4. Keep existing training history

## Customizing Training Parameters

Edit `train.py` call in SLURM scripts or run directly:

```bash
python train.py \
    --epochs 100 \           # Number of epochs
    --batch-size 64 \        # Larger batch if you have RAM
    --samples-per-image 10 \ # More augmentation
    --lr 5e-5 \             # Learning rate
    --num-workers 16 \       # Data loading workers
    --output-dir my_output   # Custom output directory
```

Full options:
```bash
python train.py --help
```

## Monitoring and Debugging

### Check job status
```bash
# List your jobs
squeue -u $USER

# Detailed job info
scontrol show job <job_id>

# Cancel a job
scancel <job_id>
```

### Check outputs while running
```bash
# Training progress
tail -f logs/full_<job_id>.out

# Errors (should be empty if all is well)
tail -f logs/full_<job_id>.err

# Check latest training plot
ls -lh outputs/*/training_history.png
```

### Common issues

**Problem: Job immediately fails**
- Check partition name is correct (`sinfo`)
- Check module names are correct (`module avail`)
- Check environment activation path

**Problem: "Permission denied" on scripts**
```bash
chmod +x train.py train_test.slurm train_full.slurm
```

**Problem: Out of memory**
- Reduce `--batch-size` in train.py call
- Reduce `--cpus-per-task` if memory per CPU is limited
- Check with: `scontrol show partition YOUR_PARTITION`

**Problem: Training very slow**
- Increase `--cpus-per-task` (more parallelism)
- Increase `--num-workers` for faster data loading
- Check if CPU threads are set: look for "Set OMP_NUM_THREADS" in log

## Understanding CPU Performance

CPU training is slower than GPU, but can be optimized:

**Speed factors:**
- **CPU cores**: More cores = faster (diminishing returns after ~64)
- **Batch size**: Larger batches better utilize CPU parallelism
- **DataLoader workers**: Parallel data loading (set to ~1/4 of CPUs)
- **PyTorch threads**: Set to match CPU cores (done automatically)

**Expected timing (approximate):**
- With 32 cores: ~10-15 min/epoch
- With 64 cores: ~5-8 min/epoch
- With 96 cores: ~3-5 min/epoch

For 50 epochs:
- 32 cores: ~8-12 hours
- 64 cores: ~4-7 hours
- 96 cores: ~2.5-4 hours

## Output Files

After training completes, check `outputs/full_training/`:

- `best_model.pt` - Best checkpoint (highest validation AUROC)
- `final_model.pt` - Final model after all epochs
- `model_config.json` - Model hyperparameters
- `training_history.png` - Loss/accuracy/AUROC plots
- `training_*.log` - Detailed training logs

## Tips

1. **Always run test job first** - Catches issues quickly
2. **Set realistic time limits** - Add 20% buffer to your estimate
3. **Use resume feature** - Safe for job interruptions
4. **Check logs regularly** - Catch problems early
5. **Download results** - Copy outputs to your local machine

```bash
# Download results (from your local machine)
scp -r user@cluster:/path/to/outputs/full_training ./results/
```

## Questions?

Check cluster documentation or contact your cluster administrator for:
- Available partitions and their limits
- Module names for Python/PyTorch
- Maximum CPU cores per job
- Queue priorities and wait times
