#!/bin/bash

#SBATCH --job-name=all_test_plus3
#SBATCH --output=../emaitzak/iker_testAll_plus3_%j.txt
#SBATCH --error=../emaitzak/iker_testAll_plus3_%j_error.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
# #SBATCH --ntasks-per-core=2

#SBATCH --mem-per-cpu=10G

#SBATCH --partition=GPU
#SBATCH --gpus=1

# Use SLURM array jobs to run in parallel
# Consider using 0-89 if you have 90 unique combinations (45 datasets * 2 pairs)
#SBATCH --array=0-2

export PYTHONPATH=$PYTHONPATH:/home/iperez/pfs/LGGM_2/Large-Graph-Generative-Model
export HYDRA_FULL_ERROR=1 # Good for debugging Hydra issues



DATASETS=("InternetTopology-15-N100" "InternetTopology-15-N10" "InternetTopology-15-N40")

# Usar el ID de tarea para determinar qué combinación ejecutar
TASK_ID=$SLURM_ARRAY_TASK_ID

# Calcular índices para cada parámetro
DS_INDEX=$((TASK_ID %  ${#DATASETS[@]} )) # Integer division naturally handles the step

# Comprobar si el ID de tarea es válido (optional if array limits are tight)
if [ $DS_INDEX -ge ${#DATASETS[@]} ]; then
    echo "ID de tarea $TASK_ID ($SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID) da como resultado DS_INDEX $DS_INDEX, que está fuera de rango para DATASETS. Saliendo."
    exit 1 # Non-zero exit for error
fi

# Extraer los parámetros específicos para esta tarea y EXPORTARLOS
export DATASET=${DATASETS[$DS_INDEX]}

# Echo con variables de bash estándar
echo "Tarea SLURM ID: $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID"
echo "Índices calculados -> PAIR_INDEX: ${PAIR_INDEX}, DS_INDEX: ${DS_INDEX}"
echo "Ejecutando con dataset: ${DATASET}"

# Construir el valor para general.name usando variables de bash
# Este valor se pasará directamente a Hydra.
CONSTRUCTED_GENERAL_NAME_test="all_uniform_plus3_${DATASET}_eval_tr"


echo "Nombre general construido: ${CONSTRUCTED_GENERAL_NAME_test}"

# Ejecutar el script de Python con las anulaciones de Hydra correctas
# Usa '${oc.env:VAR}' para que Hydra resuelva las variables de entorno.
# Las comillas simples alrededor del valor '${oc.env:VAR}' son cruciales para evitar la expansión de bash.

bnd -exec python3 main.py \
    dataset.name='${oc.env:DATASET}' \
    dataset.sample="seed" \
    general.name="${CONSTRUCTED_GENERAL_NAME_test}" \
    model.transition="uniform" \
    general.ckpt_path='../ft_all_wo_${oc.env:DATASET}---${oc.env:DATASET}-seed_uniform_plus3/checkpoints/ft_all_wo_${oc.env:DATASET}---${oc.env:DATASET}-seed_uniform_plus3/last-v1.ckpt' \
    train.batch_size=4 \
    general.setting='test2' \
    general.wandb="online"


echo "Script de Python finalizado para la tarea $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID."
