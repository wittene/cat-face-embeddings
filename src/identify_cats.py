"""
Author: Elizabeth Witten

Command-line program for identifying cats.
"""

####################
# IMPORTS
####################

import sys
import argparse
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

from embeddings.train_embeddings import CatFaceDataset, CatEmbedNN




####################
# CLASSES
####################

class BoxCoords:
  """
  Simple structure for holding bounding box coordinates.
  """
  def __init__(self, xmin, ymin, xmax, ymax):
    self.xmin = xmin
    self.ymin = ymin
    self.xmax = xmax
    self.ymax = ymax




####################
# FUNCTIONS
####################

def locate_face(image, face_cascade):
  """
  Locate cat face in the input image using Haar cascade classifier.
  
  Args:
    image (PIL Image): input image
  
  Returns:
    Bounding box coordinates of the located cat face, type BoxCoords,
    or None if no cat face is found
  """
  # Convert PIL image to opencv BGR image
  cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  
  # Detect cat faces in the image
  faces = face_cascade.detectMultiScale(cv2_image, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
  # Return bounding box, if found
  if len(faces) == 0:
    # No cat face detected
    return None
  else:
    # Extract the coordinates of the first detected cat face
    (x, y, w, h) = faces[0]
    # Add a buffer to the top and sides to estimate the ears
    buf_top = 0.3*h
    buf_sides = 0.15*w
    # Compute new coords
    xmin = max(0, x-buf_sides)
    ymin = max(0, y-buf_top)
    xmax = min(image.size[0], x+w+buf_sides)
    ymax = min(image.size[1], y+h)
    return BoxCoords(xmin, ymin, xmax, ymax)

def get_default_bbox(image):
  """
  Get default bounding box as a centered square 3/4 the length of the shortest side.
  
  Args:
    image (PIL Image): input image
  
  Returns:
    Bounding box coordinates, type BoxCoords
  """
  # Get image size
  width, height = image.size  
  # Determine the side length of the square
  side_length = min(width, height)
  side_length *= 0.75
  # Calculate left, top, right, and bottom coordinates of the bounding box
  xmin = (width - side_length) // 2
  ymin = (height - side_length) // 2
  xmax = xmin + side_length
  ymax = ymin + side_length  
  # Return the bounding box coordinates
  return BoxCoords(xmin, ymin, xmax, ymax)

def match_k(query, db, k=10, device="cpu"):
  """
  Find the top-k matches for the query embedding.
  Uses nearest-neighbor classification with cosine similarity.
  
  Args:
    query: query embedding, tensor with shape (1, n_features)
    db: database of embeddings
    k: number of top matches to return
  
  Returns:
    A list of the top-k subject ids  
  """
  
  # Compute pairwise cosine similiary scores between query and db embeddings, shape (1, db_size)
  pairwise_sim = F.cosine_similarity(query.unsqueeze(1), db.embeddings.unsqueeze(0), dim=2)
  
  # Get indices of top k nearest neighbors, shape (1, k)
  _, indices = torch.topk(pairwise_sim, k=k, dim=1)
  indices = indices.to(device)
  
  # Get top-k subject ids
  top_k = [db.subjects[i] for i in indices.view([-1])]
  return top_k




####################
# MAIN
####################

def main(argv):
  """
  Main function.
  Interactive command-line program for identifying cat faces.
  
  Example usage: python identify_cats.py "./trained_model/checkpoint.pth" "./db/db.pth" --k 10 
  """
  
  # handle any command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint_path", type=str, help="Checkpoint path to load embedding model")
  parser.add_argument("db_path", type=str, help="Path to database file")
  parser.add_argument("--use_latest", dest="use_latest", action="store_true", help="Use the latest model instead of best model")
  parser.add_argument("--k", type=int, help="How many matches to return per query")

  # parse parameters
  args = parser.parse_args()
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using device {device}")
    
  print("Loading checkpoint...")
  checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(device))
   
  print("Initializing embedding model...")
  model = CatEmbedNN().to(device)
  if args.use_latest:
    model.load_state_dict(checkpoint["latest_model_state_dict"])
  else:
    model.load_state_dict(checkpoint["best_model_state_dict"])
    
  print("Loading database...")
  db = EmbeddingsDatabase()
  db.load_dict(torch.load(args.db_path, map_location=torch.device(device)))
  db.to(device)
  face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")
  
  print("Ready!\n")
   
  to_tensor = transforms.ToTensor()
  image_extensions = [".jpg", ".jpeg", ".png"]
  while True:
    # Read in image file path from stdin
    fp = input("Enter image file path (or 'q' to quit): ")
    if fp.lower() == 'q':
      break
    if not os.path.exists(fp):
      print(f"File path {fp} does not exist... please try again!")
      continue
    # Open the image using PIL
    try:
      image = Image.open(fp)
    except:
      print(f"Image file path {fp} could not be opened... please try again!")
      continue
    # Try to identify cat face bounding box
    bbox = locate_face(image, face_cascade)
    if not bbox:
      print("* Warning: no cat face detected, attempting retrieval anyways...")
      bbox = get_default_bbox(image)
    print(f"Estimating face location at: {vars(bbox)}")
    # Apply pre-processing (CatFaceDataset.resize_with_padding --> toTensor --> add batch dimension)
    processed_img = to_tensor(CatFaceDataset.resize_with_padding(image, bbox)).to(device).unsqueeze(0)
    # Pass processed image to model to get embedding --> query
    query = model(processed_img)
    # Get matches
    top_subj = match_k(query, db, k=args.k, device=device)
    # Output results
    print("-"*25)
    print(f"| Query: {fp}")
    for i, subj in enumerate(top_subj):
      print(f"|  {i+1}:  {subj}")
    print("-"*25)
    print()
  
  should_save = input("Save any database additions? (Y/N)")
  if should_save.lower() == "y":
    save_path = input("Enter output file path (.pth file): ")    
    if not save_path.endswith(".pth"):
      save_path = f"{save_path}.pth"
    torch.save(db.to_dict(), save_path)
    print("Database saved!")
  else:
    print("Discarding changes...")
    
  print("Done!")




if __name__ == "__main__":
  main(sys.argv)