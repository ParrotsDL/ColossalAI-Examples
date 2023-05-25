
export DATA=/mnt/lustre/share/sunxiaoye.p/RunMModes/huggingfacegpt2/data/small-gpt-dataset.json
now=$(date +"%Y%m%d_%H%M%S")
jobname="GPT2_test"

# n_d=5
# # -w HOST-10-142-4-178 
# srun -p camb_mlu290 -n$n_d --gres=mlu:$n_d python train_gpt.py \
#     --config=gpt2_configs/gpt2_pp.py \
#     --host=127.0.0.1 \
#     --port=27642 \
#     --world_size=$n_d
n_d=8
w_s=32
# -w HOST-10-142-4-178 
srun -p camb_mlu290 -n$w_s --gres=mlu:$n_d  --ntasks-per-node=$n_d --exclude=HOST-10-142-4-190 --job-name=$jobname python eval_gpt.py \
    --config=gpt2_configs/gpt2_pp_eval.py \
    --host=127.0.0.1 \
    --port=27642 \
    --world_size=$w_s \
    2>&1 | tee log/test_${jobname}_${now}.log
