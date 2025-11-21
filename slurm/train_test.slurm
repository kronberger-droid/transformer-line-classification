#!/bin/bash
#SBATCH --job-name=vit_test              # Job name
#SBATCH --output=logs/test_%j.out        # Output file (%j = job ID)
#SBATCH --error=logs/test_%j.err         # Error file
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=32               # CPU cores (adjust as needed)
#SBATCH --mem=64G                        # Memory
#SBATCH --time=04:00:00                  # Max runtime (4 hours)
#SBATCH --partition=PLACEHOLDER          # CHANGE THIS: check with 'sinfo'

# ============================================================================
# SLURM Test Job for Vision Transformer Training (2 epochs)
#
# Before running:
# 1. Check available partitions: sinfo
# 2. Update --partition=PLACEHOLDER above with actual partition name
# 3. Adjust --cpus-per-task based on your needs (16, 32, 64, 96, etc.)
# 4. Create logs directory: mkdir -p logs
#
# To submit: sbatch train_test.slurm
# To monitor: squeue -u $USER
# To view output: tail -f logs/test_<job_id>.out
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

# Run training with test parameters
echo "=========================================="
echo "Starting Training (TEST MODE - 2 epochs)"
echo "=========================================="
echo ""

python train.py \
    --epochs 2 \
    --batch-size 8 \
    --samples-per-image 2 \
    --num-workers $NUM_WORKERS \
    --num-threads $SLURM_CPUS_PER_TASK \
    --output-dir outputs/test_run \
    --resume

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

# Check if outputs were created
if [ -f "outputs/test_run/best_model.pt" ]; then
    echo "SUCCESS: Model checkpoint created"
    ls -lh outputs/test_run/best_model.pt
else
    echo "WARNING: Model checkpoint not found"
fi

exit $EXIT_CODE
