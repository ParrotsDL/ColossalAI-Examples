export DATA=/mnt/lustre/share/sunxiaoye.p/RunMModes/huggingfacegpt2/data/small-gpt-dataset.json
python train_gpt.py \
    --config=gpt2_configs/gpt2_pp.py \
    --host=127.0.0.1 \
    --port=25532 \
    --world_size=${SLURM_NTASKS} \
    > outputs/output.${SLURM_PROCID}.log 2>&1