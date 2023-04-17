#!/bin/bash
#SBATCH --job-name=embd_vis
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --output=proj.%j.out
#SBATCH --error=proj.%j.err
echo "Starting..."
source activate cuda117_env
python ./src/embeddings/visualize_tsne.py \
  --image_dir "./data/split/train" \
  --annotation_scale 0.015\
  "./db/db.pth" \
  "./trained_model/tsne_sm.png"
python ./src/embeddings/visualize_tsne.py \
  --image_dir "./data/split/train" \
  --annotation_scale 0.05\
  "./db/db.pth" \
  "./trained_model/tsne_lg.png"
python ./src/embeddings/visualize_tsne.py \
  "./db/db.pth" \
  "./trained_model/tsne_noimg.png"