
export DATA=/mnt/lustre/share/sunxiaoye.p/RunMModes/huggingfacegpt2/data/small-gpt-dataset.json
now=$(date +"%Y%m%d_%H%M%S")
jobname="GPT2"

export PARROTS_ALIGN_TORCH=1
export CNCL_LOG_LEVEL=ALLOC

# export PARROTS_AUTOCAST_ENABLED=1

# n_d=5
# # -w HOST-10-142-4-178 
# srun -p camb_mlu290 -n$n_d --gres=mlu:$n_d python train_gpt.py \
#     --config=gpt2_configs/gpt2_pp.py \
#     --host=127.0.0.1 \
#     --port=27642 \
#     --world_size=$n_d

n_d=8
w_s=32
# -w HOST-10-142-4-178 --exclude=HOST-10-142-4-190
srun -p camb_mlu290 -n$w_s --gres=mlu:$n_d  --ntasks-per-node=$n_d  --job-name=$jobname python train_gpt.py \
    --config=gpt2_configs/gpt2_pp.py \
    --host=127.0.0.1 \
    --port=27645 \
    --world_size=$w_s \
    2>&1 | tee log/train_${jobname}_${now}.log
