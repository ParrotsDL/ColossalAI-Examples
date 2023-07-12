jobname="sendT"
n_d=8
w_s=8
srun -p camb_mlu290 -n$w_s --gres=mlu:$n_d --ntasks-per-node=$n_d --job-name=$jobname python send.py