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

from train_embeddings import CatFaceDataset, CatEmbedNN




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
  faces = face_cascade.detectMultiScale(cv2_image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
  # Return bounding box, if found
  if len(faces) == 0:
    # No cat face detected
    return None
  else:
    # Extract the coordinates of the first detected cat face
    (x, y, w, h) = faces[0]
    return BoxCoords(x, y, x+w, y+h)

def get_default_bbox(image):
  """
  Get default bounding box as a centered square that fills the shortest side of the image.
  
  Args:
    image (PIL Image): input image
  
  Returns:
    Bounding box coordinates, type BoxCoords
  """
  # Get image size
  width, height = image.size  
  # Determine the side length of the square
  side_length = min(width, height)  
  # Calculate left, top, right, and bottom coordinates of the bounding box
  xmin = (width - side_length) // 2
  ymin = (height - side_length) // 2
  xmax = xmin + side_length
  ymax = ymin + side_length  
  # Return the bounding box coordinates
  return BoxCoords(xmin, ymin, xmax, ymax)

def match_k(query, embeddings, subjects, k=10, device="cpu"):
  """
  Find the top-k matches for the query embedding.
  Uses nearest-neighbor classification with cosine similarity distance.
  
  Args:
    query: query embedding, tensor with shape (1, n_features)
    embeddings: database of embeddings, tensor with shape (db_size, n_features)
    subjects: list of subject ids correspond to each database embedding, list of length (db_size)
    k: number of top matches to return
  
  Returns:
    A list of the top-k subject ids  
  """
  
  # Compute pairwise cosine similiary distances between query and db embeddings, shape (1, db_size)
  pairwise_dist = 1 - F.cosine_similarity(query.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
  
  # Get indices of top k nearest neighbors, shape (1, k)
  _, indices = torch.topk(pairwise_dist, k=k, dim=1)
  indices = indices.to(device)




####################
# MAIN
####################

def main(argv):
  """
  Main function.
  Interactive command-line program for identifying cat faces.
  
  Example usage: python identify_cats.py "./out_11/checkpoint.pth" "./data/db/train_out_11.pth" --k 10 
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
  db = torch.load(args.db_path, map_location=torch.device(device))
  embeddings = db["embeddings"].to(device)
  subjects = db["subjects"]  
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
      image = Image.open(file_path)
    except:
      print(f"Image file path {fp} could not be opened... please try again!")
      continue
    # Try to identify cat face bounding box
    bbox = locate_face(image, face_cascade)
    if not bbox:
      bbox = get_default_bbox(image)
    # Apply pre-processing (CatFaceDataset.resize_with_padding --> toTensor --> add batch dimension)
    processed_img = to_tensor(CatFaceDataset.resize_with_padding(image, bbox)).to(device).unsqueeze(0)
    # Pass processed image to model to get embedding --> query
    query = model(processed_img)
    # Get matches
    top_subj = match_k(query, embeddings, subjects, k=args.k, device=device)
    # Output results
    print("-"*25)
    print(f"Query: {fp}")
    for i, subj in enumerate(top_subj):
      print(f"|  {i+1}:  {subj}")
    print("-"*25)
    print()
    
  
    
  print("Done!")




if __name__ == "__main__":
  main(sys.argv)