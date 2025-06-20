#!/bin/bash

#SBATCH --job-name=all_finetuning
#SBATCH --output=../emaitzak/iker_finetuneAll_corrected3_%j.txt
#SBATCH --error=../emaitzak/iker_finetuneAll_corrected3_%j_error.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
# #SBATCH --ntasks-per-core=2

#SBATCH --mem-per-cpu=10G

#SBATCH --partition=GPU
#SBATCH --gpus=1

# Use SLURM array jobs to run in parallel
# Consider using 0-89 if you have 90 unique combinations (45 datasets * 2 pairs)
#SBATCH --array=0-1

export PYTHONPATH=$PYTHONPATH:/home/iperez/pfs/LGGM_2/Large-Graph-Generative-Model
export HYDRA_FULL_ERROR=1 # Good for debugging Hydra issues

DATASETS=("erdos-renyi-15-M10" "erdos-renyi-15-M100" "erdos-renyi-15-M40" "erdos-renyi-150-M10" "erdos-renyi-150-M100" "erdos-renyi-150-M40" "erdos-renyi-5-M10" "erdos-renyi-5-M100" "erdos-renyi-5-M40" "erdos-renyi-50-M10" "erdos-renyi-50-M100" "erdos-renyi-50-M40" "erdos-renyi-500-M10" "erdos-renyi-500-M100" "erdos-renyi-500-M40" "InternetTopology-15-N10" "InternetTopology-15-N100" "InternetTopology-15-N40" "InternetTopology-150-N10" "InternetTopology-150-N100" "InternetTopology-150-N40" "InternetTopology-5-N10" "InternetTopology-5-N100" "InternetTopology-5-N40" "InternetTopology-50-N10" "InternetTopology-50-N100" "InternetTopology-50-N40" "InternetTopology-500-N10" "InternetTopology-500-N100" "InternetTopology-500-N40" "USA-road-15-N10" "USA-road-15-N100" "USA-road-15-N40" "USA-road-150-N10" "USA-road-150-N100" "USA-road-150-N40" "USA-road-5-N10" "USA-road-5-N100" "USA-road-5-N2" "USA-road-5-N40" "USA-road-50-N10" "USA-road-50-N100" "USA-road-50-N40" "USA-road-500-N10" "USA-road-500-N100" "USA-road-500-N40")
CKPT_PTHS=("last-v1.ckpt" "last-v2.ckpt")
GEN_NAMES=("seed_uniform_plus3" "seed_uniform")

# Usar el ID de tarea para determinar qué combinación ejecutar
TASK_ID=$SLURM_ARRAY_TASK_ID

# Calcular índices para cada parámetro
# Assumes CKPT_PTHS and GEN_NAMES have the same length and elements correspond
PAIR_INDEX=$(( TASK_ID % ${#CKPT_PTHS[@]} ))
DS_INDEX=$(( (TASK_ID / ${#GEN_NAMES[@]}) % ${#DATASETS[@]} )) # Integer division naturally handles the step

# Comprobar si el ID de tarea es válido (optional if array limits are tight)
if [ $DS_INDEX -ge ${#DATASETS[@]} ]; then
    echo "ID de tarea $TASK_ID ($SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID) da como resultado DS_INDEX $DS_INDEX, que está fuera de rango para DATASETS. Saliendo."
    exit 1 # Non-zero exit for error
fi

# Extraer los parámetros específicos para esta tarea y EXPORTARLOS
export DATASET=${DATASETS[$DS_INDEX]}
export CKPT_PTH=${CKPT_PTHS[$PAIR_INDEX]}
export GEN_NAME=${GEN_NAMES[$PAIR_INDEX]}

# Echo con variables de bash estándar
echo "Tarea SLURM ID: $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID"
echo "Índices calculados -> PAIR_INDEX: ${PAIR_INDEX}, DS_INDEX: ${DS_INDEX}"
echo "Ejecutando con dataset: ${DATASET}, checkpoint path: ../${GEN_NAME}/checkpoints/${GEN_NAME}/${CKPT_PTH}"

# Construir el valor para general.name usando variables de bash
# Este valor se pasará directamente a Hydra.
CONSTRUCTED_GENERAL_NAME="ft_all_wo_${DATASET}---${DATASET}-${GEN_NAME}"

echo "Nombre general construido: ${CONSTRUCTED_GENERAL_NAME}"
echo "Ruta de checkpoint para Hydra: ../\${oc.env:GEN_NAME}/checkpoints/\${oc.env:GEN_NAME}/\${oc.env:CKPT_PTH}"

# Ejecutar el script de Python con las anulaciones de Hydra correctas
# Usa '${oc.env:VAR}' para que Hydra resuelva las variables de entorno.
# Las comillas simples alrededor del valor '${oc.env:VAR}' son cruciales para evitar la expansión de bash.
bnd -exec python3 main.py \
    dataset.name='${oc.env:DATASET}' \
    dataset.sample="seed" \
    general.name="${CONSTRUCTED_GENERAL_NAME}" \
    model.transition="uniform" \
    general.gpus=[0] \
    train.batch_size=12 \
    train.accumulate_grad_batches=4 \
    general.setting='train_from_pretrained' \
    general.ckpt_path='../all_${oc.env:GEN_NAME}/checkpoints/all_${oc.env:GEN_NAME}/${oc.env:CKPT_PTH}' \
    train.n_epochs=600 \
    general.wandb="online"

echo "Script de Python finalizado para la tarea $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID."