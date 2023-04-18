"""
Author: Elizabeth Witten

Utilities to generate a database of embeddings.
"""

####################
# IMPORTS
####################

import sys
import argparse
import torch

# Handle relative imports
parent_module = sys.modules['.'.join(__name__.split('.')[:-1]) or '__main__']
if __name__ == '__main__' or parent_module.__name__ == '__main__':
  from train_embeddings import CatEmbedNN, CatFaceDataLoader
  from database import generate_db
else:
  from .train_embeddings import CatEmbedNN, CatFaceDataLoader
  from .database import generate_db
  
  


####################
# MAIN
####################

def main(argv):
  """
  Main function.
  Generate and save database of embeddings.
  
  Example usage: python generate_database.py "./data/split/train" "./data/labels.csv" "./trained_model/checkpoint.pth" "./db/db.pth"
  """
  
  # handle any command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("data_dir", type=str, help="Directory with data to store in database")
  parser.add_argument("labels_path", type=str, help="Path to annotation file")
  parser.add_argument("checkpoint_path", type=str, help="Checkpoint path to load embedding model")
  parser.add_argument("save_path", type=str, help="Where to save database file")
  parser.add_argument("--use_latest", dest="use_latest", action="store_true", help="Use the latest model instead of best model")


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
    
  print("Setting up data...")
  loader = CatFaceDataLoader([(args.data_dir, args.labels_path)], batch_size=1)
   
  print("Generating database...")
  generate_db(model, loader, args.save_path, device)
    
  print("Done!")




if __name__ == "__main__":
  main(sys.argv)