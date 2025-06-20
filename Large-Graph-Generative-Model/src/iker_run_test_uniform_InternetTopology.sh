#!/bin/bash

#SBATCH --job-name=LGGM_testing_InternetTopology
#SBATCH --output=../emaitzak/iker_run_test_uniform_InternetTopology_%j.txt
#SBATCH --error=../emaitzak/iker_run_test_uniform_InternetTopology_%j_error.txt

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


bnd -exec python3 main.py dataset.name="InternetTopology" dataset.sample="seed" general.name="all_uniform_InternetTopology_eval" model.transition="uniform" general.ckpt_path='../ft_all_wo_InternetTopology---InternetTopology-seed_uniform/checkpoints/ft_all_wo_InternetTopology---InternetTopology-seed_uniform/last-v1.ckpt' train.batch_size=4 general.setting='test' general.wandb="online"
        

        