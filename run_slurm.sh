#!/bin/bash
dataset=market1501
#dataset=duke
#dataset=mars
#dataset=DukeMTMC-VideoReID

batchSize=16
size_penalty=0.003
merge_percent=0.05

logs=logs/$dataset

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

root_dir=/mnt/lustre/$(whoami)
project_dir=$root_dir/projects/Bottom-up-Clustering-Person-Re-identification
data_dir=$root_dir/fedreid_data/Market/pytorch/

export PYTHONPATH=$PYTHONPATH:${project_dir}

srun -u --partition=Sensetime --job-name=buc \
    -n1 --gres=gpu:1 --ntasks-per-node=1 \
    python ${project_dir}/run.py --dataset $dataset --logs_dir $logs --data_dir  \
              -b $batchSize --size_penalty $size_penalty -mp $merge_percent | tee log/train-${now}.log &