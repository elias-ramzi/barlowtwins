#!/bin/bash
#SBATCH --job-name=uINAT     # job name
#SBATCH --nodes=2
#SBATCH --ntasks=8                   # number of MP tasks
#SBATCH --ntasks-per-node=4          # number of MPI tasks per node
#SBATCH --gres=gpu:4                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=logs/%A_%a.out # output file name
#SBATCH --error=logs/%A_%a.err  # error file name
#SBATCH --array=0

set -x
cd $WORK/barlowtwins/

module purge
module load pytorch-gpu/py3/1.11.0

export TORCH_HOME=${SCRATCH}/pretrained_models
export TMPDIR=$JOBSCRATCH
export TQDM_DISABLE='yes'


srun python main.py \
'$SCRATCH/Stanford_Online_Products' \
--dataset sop \
--checkpoint-dir '$SCRATCH/experiments/barlowtwins/sop_resnet50' \
--batch-size 1024 \
--learning-rate-weights 0.1 \
--epochs 600 \
--transform randaugment
