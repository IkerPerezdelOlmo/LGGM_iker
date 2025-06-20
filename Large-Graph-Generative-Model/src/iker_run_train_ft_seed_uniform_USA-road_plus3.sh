#!/bin/bash

#SBATCH --job-name=LGGM_finetuning_USA-road_plus3
#SBATCH --output=../emaitzak/iker_run_finetune_uniform_USA-road_plus3_%j.txt
#SBATCH --error=../emaitzak/iker_run_finetune_uniform_USA-road_plus3_%j_error.txt

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




bnd -exec  python3 main.py dataset.name="USA-road" dataset.sample="seed" general.name="ft_all_wo_USA-road---USA-road-seed_uniform_plus3" model.transition="uniform" general.gpus=[0] train.batch_size=12 train.accumulate_grad_batches=4 general.setting='train_from_pretrained' general.ckpt_path='../all_seed_uniform_plus3/checkpoints/all_seed_uniform_plus3/last-v1.ckpt' train.n_epochs=600 general.wandb="online"



