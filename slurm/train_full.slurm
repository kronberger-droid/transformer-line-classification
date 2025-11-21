#!/bin/bash
#SBATCH --job-name=vit_full              # Job name
#SBATCH --output=logs/full_%j.out        # Output file (%j = job ID)
#SBATCH --error=logs/full_%j.err         # Error file
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=64               # CPU cores (adjust as needed)
#SBATCH --mem=128G                       # Memory
#SBATCH --time=72:00:00                  # Max runtime (72 hours = 3 days)
#SBATCH --partition=PLACEHOLDER          # CHANGE THIS: check with 'sinfo'

# ============================================================================
# SLURM Full Training Job for Vision Transformer (50 epochs)
#
# Before running:
# 1. Check available partitions: sinfo
# 2. Update --partition=PLACEHOLDER above with actual partition name
# 3. Adjust --cpus-per-task based on your needs (32, 64, 96, etc.)
# 4. Create logs directory: mkdir -p logs
# 5. Optionally: run test job first (train_test.slurm)
#
# To submit: sbatch train_full.slurm
# To monitor: squeue -u $USER
# To view output: tail -f logs/full_<job_id>.out
# To cancel: scancel <job_id>
#
# This script can resume from the test checkpoint if --resume is used!
# ============================================================================

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Load required modules (adjust based on your cluster)
# Common examples:
# module load python/3.10
# module load anaconda3
# module load pytorch
echo "Loading modules..."
# UNCOMMENT AND ADJUST AS NEEDED:
# module load python/3.10

# Activate virtual environment
# ADJUST THIS PATH TO YOUR ENVIRONMENT:
echo "Activating Python environment..."
# source /path/to/your/venv/bin/activate
# OR for conda:
# conda activate your_env_name

# Check Python and PyTorch
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch {torch.__version__}')"
echo ""

# Set PyTorch threading for CPU optimization
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Set OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Set MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo ""

# Calculate DataLoader workers (1/4 of CPUs)
NUM_WORKERS=$((SLURM_CPUS_PER_TASK / 4))
if [ $NUM_WORKERS -lt 1 ]; then
    NUM_WORKERS=1
fi
echo "Using $NUM_WORKERS DataLoader workers"
echo ""

# Run full training
echo "=========================================="
echo "Starting Full Training (50 epochs)"
echo "=========================================="
echo ""

python train.py \
    --epochs 50 \
    --batch-size 32 \
    --samples-per-image 5 \
    --lr 1e-4 \
    --weight-decay 0.05 \
    --num-workers $NUM_WORKERS \
    --num-threads $SLURM_CPUS_PER_TASK \
    --output-dir outputs/full_training \
    --resume

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

# Check outputs
echo ""
echo "Checking outputs..."
if [ -f "outputs/full_training/best_model.pt" ]; then
    echo "SUCCESS: Best model checkpoint created"
    ls -lh outputs/full_training/best_model.pt
fi

if [ -f "outputs/full_training/final_model.pt" ]; then
    echo "SUCCESS: Final model saved"
    ls -lh outputs/full_training/final_model.pt
fi

if [ -f "outputs/full_training/training_history.png" ]; then
    echo "SUCCESS: Training plots created"
    ls -lh outputs/full_training/*.png
fi

if [ -f "outputs/full_training/model_config.json" ]; then
    echo "SUCCESS: Model config saved"
    cat outputs/full_training/model_config.json
fi

exit $EXIT_CODE
