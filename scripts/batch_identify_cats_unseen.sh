#!/bin/bash
#SBATCH --job-name=embd_id
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --output=proj.%j.out
#SBATCH --error=proj.%j.err
echo "Starting..."
source activate cuda117_env
python ./src/identify_cats.py --k 10 "./trained_model/checkpoint.pth" "./db/db.pth" < ./test/test_cats_unseen.txt