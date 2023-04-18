"""
Author: Elizabeth Witten

Utilities to visualize a database of embeddings in 2D using t-SNE.
"""

####################
# IMPORTS
####################

import sys
import os
import argparse
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# Handle relative imports
parent_module = sys.modules['.'.join(__name__.split('.')[:-1]) or '__main__']
if __name__ == '__main__' or parent_module.__name__ == '__main__':
    from database import EmbeddingsDatabase
else:
    from .database import EmbeddingsDatabase




####################
# FUNCTIONS
####################

def visualize_tsne(db, save_path, image_dir=None, annotation_scale=0.01):
  """
  Visualizes t-SNE embeddings with associated images.
  
  Args:
    db: EmbeddingsDatabase
    save_path: where to save t-SNE image
    image_dir: if provided, where to search for original images to add annotations
  """
  
  # Get embeddings and subjects
  embeddings = db.embeddings.numpy()
  subjects = db.subjects

  # Perform t-SNE to get 2D space
  tsne = TSNE(n_components=2, learning_rate="auto", metric="cosine", init="pca", random_state=42)
  X_tsne = tsne.fit_transform(embeddings)

  # Plot t-SNE embeddings
  fig = plt.figure()
  if image_dir:
    # Use lighter markers
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], marker="o", c="aliceblue", edgecolors="silver")
  else:
    # Use dark solid markers
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], marker="o", c="darkslategray", s=25)

  # If provided, add images as annotations
  if image_dir:
    # For each point...
    for i in range(X_tsne.shape[0]):
      # Load the image
      subj_dir = os.path.join(image_dir, subjects[i])
      subj_file = os.path.join(subj_dir, os.listdir(subj_dir)[0])
      image = np.array(Image.open(subj_file), dtype='float64')
      # Normalize to [0, 1]
      image /= 255.0
      # Create an AnnotationBbox with the image as the annotation
      imagebox = OffsetImage(image, zoom=annotation_scale)
      # Place annotation
      xy = X_tsne[i]
      ab = AnnotationBbox(imagebox, xy, frameon=False)
      plt.gca().add_artist(ab)

  # Finalize and save plot
  plt.xlabel('t-SNE Dimension 1')
  plt.ylabel('t-SNE Dimension 2')
  plt.title('t-SNE Visualization of Cat Face Embeddings')
  fig.savefig(save_path, dpi=1000)
  plt.close(fig)




####################
# MAIN
####################

def main(argv):
  """
  Main function.
  Visualize embedding space in 2D.
  
  Example usage: python visualize_tsne.py "./db/trained_model.pth" "./trained_model/tsne.png" --image_dir "./data/split/train"
  """
  
  # handle any command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("db_path", type=str, help="Path to database file")
  parser.add_argument("save_path", type=str, help="Where to save visualization")
  parser.add_argument("--image_dir", type=str, default=None, help="If provided, original images to annotate visualization")
  parser.add_argument("--annotation_scale", type=float, default=0.01, help="Scale original image for annotation, ignored when image_dir is not provided")

  # parse parameters
  args = parser.parse_args()
    
  print("Loading database...")
  db = EmbeddingsDatabase()
  db.load_dict(torch.load(args.db_path, map_location=torch.device("cpu")))
   
  print("Generating t-SNE visualization...")
  visualize_tsne(db, args.save_path, image_dir=args.image_dir, annotation_scale=args.annotation_scale)
    
  print("Done!")




if __name__ == "__main__":
  main(sys.argv)