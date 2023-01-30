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
#SBATCH --array=0-5

set -x
cd $WORK/barlowtwins/

module purge
module load pytorch-gpu/py3/1.11.0

export TORCH_HOME=${SCRATCH}/pretrained_models
export TMPDIR=$JOBSCRATCH
export TQDM_DISABLE='yes'

lr[0]=0.1
bs[0]=1024
name[0]=$SCRATCH/experiments/barlowtwins/sop_resnet50_bs1024

lr[1]=0.4
bs[1]=1024
name[1]=$SCRATCH/experiments/barlowtwins/sop_resnet50_bs1024_lr0.4

lr[2]=0.4
bs[2]=512
name[2]=$SCRATCH/experiments/barlowtwins/sop_resnet50_bs512_lr0.4

lr[3]=0.1
bs[3]=512
name[3]=$SCRATCH/experiments/barlowtwins/sop_resnet50_bs512_lr0.1

lr[4]=0.1
bs[4]=256
name[4]=$SCRATCH/experiments/barlowtwins/sop_resnet50_bs256_lr0.1

lr[5]=0.4
bs[5]=256
name[5]=$SCRATCH/experiments/barlowtwins/sop_resnet50_bs256_lr0.4



srun python main.py \
$SCRATCH/Stanford_Online_Product \
--dataset cub200 \
--pretrained \
--checkpoint-dir ${name[${SLURM_ARRAY_TASK_ID}]} \
--batch-size ${bs[${SLURM_ARRAY_TASK_ID}]} \
--learning-rate-weights ${lr[${SLURM_ARRAY_TASK_ID}]} \
--epochs 200 \
--transform base
