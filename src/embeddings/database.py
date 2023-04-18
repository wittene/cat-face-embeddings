"""
Author: Elizabeth Witten

Utilities to manage a database of embeddings.
"""

####################
# IMPORTS
####################

import sys
import argparse
import torch




####################
# CLASSES
####################

class EmbeddingsDatabase:
  """
  Simple storage class for database.
  """
    
  def __init__(self, embeddings=None, subjects=None):
    """
    Initializes a database.
    
    Args:
      embeddings: tensor of embeddings, shape (db_size, n_features)
      subjects: subject ids, list of length db_size
    """
    self.embeddings = [] if embeddings is None else embeddings
    self.subjects = [] if subjects is None else subjects

  def to_dict(self):
    """
    Return:
      database as dictionary
    """
    return {
      "embeddings": self.embeddings,
      "subjects": self.subjects
    }
 
  def load_dict(self, d):
    """
    Load state from dictionary.
    
    Args:
      d: dictionary to load
    """
    self.embeddings = d["embeddings"]
    self.subjects = d["subjects"]
  
  def to(self, device):
    """
    Send database tensors to device.
    
    Args:
      device: device
    """
    self.embeddings = self.embeddings.to(device)
  
  def concat(self, embeddings, subjects):
    """
    Concatenate entries to database.
    
    Args:
      embeddings: tensor of embeddings, shape (entries_size, n_features)
      subjects: subject ids, list of length entries_size
    """
    self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)
    self.subjects += subjects

    


####################
# FUNCTIONS
####################

def generate_db(model, loader, save_path=None, device="cpu"):
  """
  Generate a database from a dataloader.
  
  Args:
    model: model to produce embeddings
    loader: data loader that points to the data to embed
    save_path: if provided, where to save the database file
    device: device
  """
    
  # Gather embeddings
  embeddings = []
  classids = []
  model.eval()
  with torch.no_grad():
    for X, y in loader:
      X, y = X.to(device), y.to(device)
      embd = model(X)
      embeddings.append(embd)
      classids.append(y)

  # Merge lists of tensors into single tensor
  embeddings = torch.cat(embeddings, dim=0)
  classids = torch.cat(classids, dim=0)
  
  # Convert classids to subject ids
  subjects = [loader.classes[classids[i]] for i in range(classids.size(0))]
  
  # Save and return
  db = EmbeddingsDatabase(embeddings=embeddings, subjects=subjects)
    
  if save_path:
    if not save_path.endswith(".pth"):
      save_path = f"{save_path}.pth"
    torch.save(db.to_dict(), save_path)
    print("Database saved!")
    
  return db