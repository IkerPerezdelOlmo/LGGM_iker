#!/bin/bash

#SBATCH --job-name=LGGM_finetuning
#SBATCH --output=../emaitzak/iker_run_finetune_uniform_erdos-renyi_%j.txt
#SBATCH --error=../emaitzak/iker_run_finetune_uniform_erdos-renyi_%j_error.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
# #SBATCH --ntasks-per-core=2

#SBATCH --mem-per-cpu=10G

#SBATCH --partition=GPU
#SBATCH --gpus=1

# Use SLURM array jobs to run in parallel
#SBATCH --array=1-1

export PYTHONPATH=$PYTHONPATH:/home/iperez/pfs/LGGM_2/Large-Graph-Generative-Model
export HYDRA_FULL_ERROR=1




bnd -exec  python3 main.py dataset.name="erdos-renyi" dataset.sample="seed" general.name="ft_all_wo_erdos-renyi---erdos-renyi-seed_uniform" model.transition="uniform" general.gpus=[0] train.batch_size=12 train.accumulate_grad_batches=4 general.setting='train_from_pretrained' general.ckpt_path='../all_seed_uniform/checkpoints/all_seed_uniform/last-v2.ckpt' train.n_epochs=600 general.wandb="online"
