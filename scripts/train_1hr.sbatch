#!/usr/bin/env bash
#SBATCH --job-name=nanobrain
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=3:00:00
#SBATCH --partition=n
#SBATCH --account=training
#SBATCH --output=slurms/slurm-%j.out
#SBATCH --error=slurms/slurm-%j.err

set -euo pipefail

export BRAINMARKS_SMRI_DATA="${BRAINMARKS_SMRI_DATA:-data}"

OUT_DIR="output"
NAME="vits_1hr"
RUN_DIR="${OUT_DIR}/${NAME}"

# run training with a fixed wall time budget
# timeout exits 124 when it stops training, which is expected
status=0
timeout --kill-after=2m "1h" \
    uv run --no-sync python -m nanobrain.train \
    --overrides \
    name="${NAME}" \
    output_dir="${OUT_DIR}" \
    || status=$?
if [[ $status -ne 0 && $status -ne 124 ]]; then
    exit $status
fi

# run all brainmarks-smri evals, results in ${RUN_DIR}/eval/<task>.json
TASKS=$(uv run --no-sync python -c "from brainmarks_smri.tasks import list_tasks; print(*list_tasks())")
for task in ${TASKS}; do
    uv run --no-sync python -m brainmarks_smri vit_mae_3d "${task}" \
        --overrides \
        "imports=[nanobrain.model]" \
        model_kwargs.checkpoint="${RUN_DIR}/checkpoint-last.pth" \
        output_dir="${RUN_DIR}" \
        name=eval
done
