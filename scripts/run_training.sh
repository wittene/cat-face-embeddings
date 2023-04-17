#!/bin/bash
#SBATCH --job-name=cat_embd
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --output=./trained_model/proj.%j.out
#SBATCH --error=./trained_model/proj.%j.err
echo "Starting..."
source activate cuda117_env
python ./src/train_embeddings.py \
 --debug \
 --batch_size 128 \
 --lr 0.01 \
 --loss_margin 0.1 \
 --n_epochs 100 \
 --logging_freq 5 \
 --save_dir "./trained_model"\
 --augmented_data "./data/by_id_augmented","./data/by_id_augmented/labels.csv" \
 "./data/split/train" "./data/split/valid" "./data/split/test" "./data/labels.csv"