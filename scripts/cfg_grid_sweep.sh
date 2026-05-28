#!/bin/bash
# Sequential CFG grid sweep on GPU 1.
# Teacher (3-fwd CFG path) + OLD gcbc + min=5, 100 seq each.
# Usage: bash scripts/cfg_grid_sweep.sh <ctx_list> <prompt_list> <tag>
#   e.g. bash scripts/cfg_grid_sweep.sh "1.0 3.0 5.0" "3.0 7.0 11.0" coarse
set -euo pipefail

CTX_LIST="${1}"
PROMPT_LIST="${2}"
TAG="${3:-grid}"

cd "$(dirname "$0")/.."
source .venv/bin/activate

for ctx in $CTX_LIST; do
  for prompt in $PROMPT_LIST; do
    model_id="cfg_${TAG}_ctx${ctx}_prompt${prompt}_100"
    log="logs/eval_${model_id}.log"
    echo "=== $model_id ===" | tee -a "logs/cfg_${TAG}_summary.log"
    CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_calvin.py \
      --mode taksie \
      --flowdit_ckpt models/calvin_ft_subgoal_rae/model.pt \
      --gcbc_ckpt models/gcbc_abcd/gcbc_step400000.pt \
      --config config/ablation/timestep_distribution/calvin_subgoal_rae.yaml \
      --env_cfg data/calvin/task_ABC_D/validation \
      --model_id "$model_id" \
      --num_sequences 100 --device 0 \
      --act_pred_horizon 5 \
      --delta_high 0.90 --min_per_frame 5 --max_per_frame 20 \
      --flowdit_num_steps 4 \
      --prompt_cfg_scale "$prompt" --context_cfg_scale "$ctx" \
      > "$log" 2>&1
    # Extract final avg from log
    final=$(tr '\r' '\n' < "$log" | grep -oE "\[100/100\] result=[0-9] \| avg_len=[0-9.]+" | tail -1)
    echo "  ctx=$ctx prompt=$prompt -> $final" | tee -a "logs/cfg_${TAG}_summary.log"
  done
done

echo "Sweep done. Summary in logs/cfg_${TAG}_summary.log"