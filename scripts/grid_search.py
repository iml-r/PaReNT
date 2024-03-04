import os
import subprocess
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--arch_name', type=str, help='Architecture name')
parser.add_argument('--units', type=int, default=128, help='Number of units')
parser.add_argument('--test_mode', type=bool, default=False, help='Use only one hundredth of the available data')
args = parser.parse_args()

#batch_sz_lst =          [64, 32, 16, 8]*2
units_lst =              [1024, 2048,  2048, 1024]
# #char_embeddings_lst =   [64, 128, 64, 128]*2
# lr_lst =                 [0.001, 0.0005, 0.0001, 0.0001]*2
transblocks =            [1, 1, 1, 1]
dropout_lst =            [0.3, 0.5, 0.5, 0.3]
recurrent_dropout_lst =  [0.2, 0.3, 0.2, 0.3]
# use_lang_embed =         [1, 1, 1, 1] + [1, 1, 1, 1]
# optimizers =             ["Adam"]*8 #+ ["SGD"]*4
attention_units_lst =        [64, 128, 512, 128]
# l_lst =                      [None, "l1", "l2", "l1_l2"] + ["l1_l2", "l2", "l1", None]

for (units,
     transblock,
     attention_units,
     dropout,
     recurrent_dropout) in zip(units_lst,
                               transblocks,
                               attention_units_lst,
                               dropout_lst,
                               recurrent_dropout_lst):

    #name = "_".join([str(i) for i in [transblock, dropout, recurrent_dropout, use_lang_embed, semantic_warmup, optimizer]])
    command = f"""#!/bin/bash
#SBATCH -J {args.arch_name}{units}{transblock}{attention_units}			  # name of job
#SBATCH -p gpu-ms,gpu-troja					  # name of partition or queue (default=cpu-troja)
#SBATCH -o {args.arch_name}{units}{transblock}{attention_units}.out			  # name of output file for this submission script
#SBATCH -e {args.arch_name}{units}{transblock}{attention_units}.err			  # name of output file for this submission script
    
#SBATCH --gres=gpu:1                          # snumber of GPUs to request (default 0)
#SBATCH --mem=64G                             # request n gigabytes memory (per node, default depends on node)
    
python3 PaReNT_tensorflow.py --arch_name {args.arch_name} --units {units} --cluster 1 --test_mode 0 --transblocks {transblock} --dropout {recurrent_dropout} --recurrent_dropout {dropout} --use_lang_embed 1 --semantic_warmup 0 --optimizer Adam --l l1 --batch_sz 64 --frac_mode 1 --attention_units {attention_units}
"""
    file = open("job.sh", "w")
    file.write(command)
    file.close()


    subprocess.run("sbatch job.sh".split())

    print("Running:", command)

check_queue = "squeue -u svoboda"
subprocess.run(check_queue.split())

time.sleep(5)
subprocess.run(check_queue.split())

time.sleep(15)
subprocess.run(check_queue.split())

time.sleep(120)
subprocess.run(check_queue.split())

# parser = argparse.ArgumentParser()
# parser.add_argument('--cluster', type=bool, default=True, help='Cluster or home?')
# parser.add_argument('--batch_sz', type=int, default=256, help='batch size')
# parser.add_argument('--epochs', type=bool, default=10, help='No. of epochs')
# parser.add_argument('--units', type=int, default=128, help='No. of units in most layers')
# parser.add_argument('--char_embeddings', type=int, default=8, help='Dim of character embeddings')
# parser.add_argument('--lr', type=int, default=1e-3, help='Dim of character embeddings')
# parser.add_argument('--test_mode', type=bool, default=False, help='Use only one hundredth of the abailable data')
# args = parser.parse_args()
