n_g=8
w_s=32
srun -p camb_mlu290 -n$w_s --gres=mlu:$n_g --exclude=HOST-10-142-4-190 sh run2_gpt.sh