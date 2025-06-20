#!/bin/bash

#SBATCH --job-name=all_finetuning
#SBATCH --output=../emaitzak/iker_finetuneAll_%j.txt
#SBATCH --error=../emaitzak/iker_finetuneAll_%j_error.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
# #SBATCH --ntasks-per-core=2

#SBATCH --mem-per-cpu=10G

#SBATCH --partition=GPU
#SBATCH --gpus=1

# Use SLURM array jobs to run in parallel
#SBATCH --array=0-91

export PYTHONPATH=$PYTHONPATH:/home/iperez/pfs/LGGM_2/Large-Graph-Generative-Model
export HYDRA_FULL_ERROR=1

DATASETS=("erdos-renyi-15-M10" "erdos-renyi-15-M100" "erdos-renyi-15-M40" "erdos-renyi-150-M10" "erdos-renyi-150-M100" "erdos-renyi-150-M40" "erdos-renyi-5-M10" "erdos-renyi-5-M100" "erdos-renyi-5-M40" "erdos-renyi-50-M10" "erdos-renyi-50-M100" "erdos-renyi-50-M40" "erdos-renyi-500-M10" "erdos-renyi-500-M100" "erdos-renyi-500-M40" "InternetTopology-15-N10" "InternetTopology-15-N100" "InternetTopology-15-N40" "InternetTopology-150-N10" "InternetTopology-150-N100" "InternetTopology-150-N40" "InternetTopology-5-N10" "InternetTopology-5-N100" "InternetTopology-5-N40" "InternetTopology-50-N10" "InternetTopology-50-N100" "InternetTopology-50-N40" "InternetTopology-500-N10" "InternetTopology-500-N100" "InternetTopology-500-N40" "USA-road-15-N10" "USA-road-15-N100" "USA-road-15-N40" "USA-road-150-N10" "USA-road-150-N100" "USA-road-150-N40" "USA-road-5-N10" "USA-road-5-N100" "USA-road-5-N2" "USA-road-5-N40" "USA-road-50-N10" "USA-road-50-N100" "USA-road-50-N40" "USA-road-500-N10" "USA-road-500-N100" "USA-road-500-N40")
CKPT_PTHS=("last-v1.ckpt" "last-v2.ckpt")
GEN_NAMES=("seed_uniform_plus3" "seed_uniform")

# Usar el ID de tarea para determinar qué combinación ejecutar
TASK_ID=$SLURM_ARRAY_TASK_ID

# Calcular índices para cada parámetro
PAIR_INDEX=$(( TASK_ID % ${#CKPT_PTHS[@]} ))
DS_INDEX=$(( (TASK_ID / ${#GEN_NAMES[@]}) % ${#DATASETS[@]} ))

# Comprobar si el ID de tarea es válido
if [ $DS_INDEX -ge ${#DATASETS[@]} ]; then
    echo "ID de tarea $TASK_ID excede el número de combinaciones. Saliendo."
    exit 0
fi


# Extraer los parámetros específicos para esta tarea
DATASET=${DATASETS[$DS_INDEX]}
CKPT_PTH=${CKPT_PTHS[$PAIR_INDEX]}
GEN_NAME=${GEN_NAMES[$PAIR_INDEX]}


echo "Ejecutando con dataset: ${DATASET}, checkpoint path: ...${CKPT_PTH}, general name: ...${GEN_NAME}"

bnd -exec  python3 main.py dataset.name=${oc.env:DATASET} dataset.sample="seed" general.name="ft_all_wo_${oc.env:DATASET}---${oc.env:DATASET}-${oc.env:GEN_NAME}"  model.transition="uniform" general.gpus=[0] train.batch_size=12 train.accumulate_grad_batches=4 general.setting='train_from_pretrained' general.ckpt_path='../${oc.env:GEN_NAME}/checkpoints/${oc.env:GEN_NAME}/${oc.env:CKPT_PTH}' train.n_epochs=300 general.wandb="online"

