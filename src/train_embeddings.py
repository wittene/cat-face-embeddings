"""
Author: Elizabeth Witten

Train cat face embeddings.
"""

####################
# IMPORTS
####################

import enum
import os
import sys
import argparse
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

import torchvision

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image




####################
# CLASSES (data)
####################


class DataParams:
  """
  Stores dataset parameters.
  """
  
  def __init__(self, 
               test_dir,
               train_dir,
               valid_dir,
               labels_path,
               augmented_data = []):    
    """
    Initializes dataset parameters.

    Args:
      test_dir: directory to LOAD test data
      train_dir: directory to LOAD training data
      valid_dir: directory to LOAD validation data
      labels_path: path to labels file, csv with bounding box annotations
      augmented_data: list of (data_dir, labels_path) tuples
    """
    self.test_dir = test_dir
    self.train_dir = train_dir
    self.valid_dir = valid_dir
    self.labels_path = labels_path
    self.augmented_data = augmented_data
    self.test_files = [(test_dir, labels_path)]
    self.train_files = [(train_dir, labels_path)] + augmented_data
    self.valid_files = [(valid_dir, labels_path)]


class CatFaceDataset(torchvision.datasets.ImageFolder):
  """
  Extends ImageFolder, with added pre-processing.
  """

  def resize_with_padding(image, label, size=224):
    """
    Resize an image to square by maintaining aspect ratio and padding the short
    side. Padding helps maintain spatial information and keeps the cat face
    centered.

    Args:
      image: image to process, type PIL
      label: object containing the face bounding box points
      size: resized square size, default is 224 for GoogLeNet
    """
    # Get image cropped to face
    face_crop = image.crop((label.xmin, label.ymin, label.xmax, label.ymax))
    # Compute aspect ratio and resize longest side to fit
    aspect_ratio = float(face_crop.width) / float(face_crop.height)
    if aspect_ratio > 1:
        new_width = size
        new_height = int(size / aspect_ratio)
    else:
        new_height = size
        new_width = int(size * aspect_ratio)
    face_crop.thumbnail((new_width, new_height), Image.BILINEAR)
    # Add 0 padding to short size to make image square
    padded_image = Image.new("RGB", (size, size), (0, 0, 0))
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    padded_image.paste(face_crop, (left, top))
    return padded_image

  def __init__(self, data_dir, label_path, transform=torchvision.transforms.ToTensor()):
    """
    Initialize an ImageFolder, plus labels.

    Args:
      data_dir: root for ImageFolder
      label_path: path to csv file with annotations
    """
    super(CatFaceDataset, self).__init__(data_dir, transform=transform)
    self.labels = pd.read_csv(label_path)
    self.label_dict = {label["filename"]: label for _, label in self.labels.iterrows()}  # dict for fast lookup
  
  def __getitem__(self, index):
    """
    Override __getitem__ in order to apply more pre-processing.
    Implementation based on https://github.com/pytorch/vision/blob/main/torchvision/datasets/folder.py
    """
    # Load
    path, target = self.imgs[index]
    sample = self.loader(path)
    label = self.label_dict[os.path.basename(path)]
    # Do transforms that require annotation data 
    sample = CatFaceDataset.resize_with_padding(sample, label)
    # Do transforms that only require image data
    if self.transform is not None:
        sample = self.transform(sample)
    if self.target_transform is not None:
        target = self.target_transform(target)
    # Return image, subject id
    return sample, target


class CatFaceDataLoader(DataLoader):
  """
  A wrapper for a torch DataLoader that points to one
  or more folders containing cat face images. 
  Underlying dataset type is CatFaceDataset.
  """

  def __init__(self, data_files, batch_size=32, shuffle=True):
    """
    Initalize the data loader.

    Args:
      data_files: list of (data_dir, label_path) tuples, where
                  data_dirs: path to data directories
                  label_path: path to annotation file
    """
    datasets = []
    for ddir, lpath in data_files:
      datasets.append(CatFaceDataset(ddir, lpath))      
    dataset = ConcatDataset(datasets)
    self.classes = datasets[0].classes
    super(CatFaceDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle)




####################
# CLASSES (model)
####################


class CatEmbedNN(nn.Module):
  """
  Model for learning cat face embeddings.
  Built on the pre-trained GoogLeNet and uses transfer learning.
  The final layer of GoogLeNet is replaced and trained to generate
  an embedding.
  """

  def __init__(self, feature_dim=256, freeze_layers=True):
    """
    Initialize the network.

    Args:
      feature_dim: size of the embedding vector
      freeze_layers: set true to freeze weights from pre-trained GoogLeNet 
    """
    super(CatEmbedNN, self).__init__()
    self.googlenet = torchvision.models.googlenet(weights="IMAGENET1K_V1")
    if freeze_layers:
      # Freeze all the pre-trained layers
      for param in self.googlenet.parameters():
        param.requires_grad = False
    # Replace the last fully connected layer to get feature embedding
    self.googlenet.fc = nn.Linear(1024, feature_dim)
    self.feature_dim = feature_dim
  
  def forward(self, x):
    """
    Forward pass through GoogLeNet.
    """
    x = self.googlenet(x)
    return x
  



####################
# CLASSES (train)
####################


class TrainingParams:
  """
  Stores training parameters.
  """
  
  def __init__(self, 
               seed = 42,
               batch_size = 64,
               lr = 0.01,
               loss_margin = 0.1,
               n_epochs = 50,
               checkpoint_path = None,
               save_dir = "/",
               logging_freq = 25,
               debug = False,
               device = "cuda" if torch.cuda.is_available() else "cpu"):    
    """
    Initializes training parameters.
    All parameters are optional and will be populated with defaults.

    Args:
      seed: random seed for torch repeatability
      batch_size: train/test batch size
      lr: learning rate
      loss_margin: margin for triplet margin loss
      n_epochs: number of training epochs
      checkpoint_path: if provided, file to LOAD a model checkpoint from
      save_dir: directory to SAVE all output (figures, trained model)
      logging_freq: how often to output training logs
      debug: whether to print out detailed output
      device: e.g. cuda or cpu
    """
    self.seed = seed
    self.batch_size = batch_size
    self.lr = lr
    self.loss_margin = loss_margin
    self.n_epochs = n_epochs
    self.checkpoint_path = checkpoint_path
    self.save_dir = save_dir
    self.logging_freq = logging_freq
    self.debug = debug
    self.device = device
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)


class Logger:
  """
  Tracks and displays performance during training.
  """

  def __init__(self, params, train_loader, valid_loader = None):
    """
    Constructs a logger.

    Args:
      params: TrainingParams
      train_loader: dataloader for training set
      valid_loader: dataloader for validation set
    """
    # for computing performance
    self.curr_epoch = 0
    self.params = params
    self.train_size = len(train_loader.dataset)
    self.valid_size = 0 if valid_loader is None else len(valid_loader.dataset)
    self.valid_batches = 0 if valid_loader is None else len(valid_loader)
    # for tracking performance
    self.train_counter = []
    self.valid_counter = []
    self.train_losses = []
    self.valid_metrics = []
  
  def log_train(self, batch, loss):
    """
    Log training performance, to be called after each training batch. 
    Only logs based on logging_freq parameter.

    Args:
      batch: current batch id (starting at 0)
      loss: training loss
    """
    if batch % self.params.logging_freq == 0:
      loss, current = loss.item(), (batch + 1) * self.params.batch_size
      if self.params.debug:
        print(f"Loss: {loss:>8f}  [{current:>5d}/{self.train_size:>5d}]")
      self.train_counter.append(current + ((self.curr_epoch-1)*self.train_size))
      self.train_losses.append(loss)
  
  def log_valid(self, metrics):
    """
    Log validation performance, to be called at the end of each epoch.

    Args:
      metrics: validation metrics
    """
    if self.valid_size == 0: return
    if self.params.debug:
      print("Validation metrics:")
      for m, val in metrics.items():
        print(f"|  {m}: {val}")
      print()
    self.valid_counter.append(self.curr_epoch * self.train_size)
    self.valid_metrics.append(metrics)
  
  def next_epoch(self):
    """
    Increment epoch that logger is tracking.
    """
    self.curr_epoch += 1
    print(f"Epoch {self.curr_epoch}\n-------------------------------")
  
  def plot_performance(self):
    """
    Plot and save a training performance graph using the values stored each call
    to log_train.
    """
    fig = plt.figure()
    plt.plot(self.train_counter, self.train_losses, color="blue")
    plt.xlabel("Training Examples Seen")
    plt.ylabel("Triplet Margin Loss (with Cosine Similarity Distance)")
    plt.title("Model Training Performance")
    fig.savefig(os.path.join(self.params.save_dir, "Performance.png"), dpi=200)
    plt.close(fig)


class OnlineTripletMining(nn.Module):
  """
  Custom loss function using online triplet mining, with triplet margin loss
  and cosine similarity distance.
  """

  def __init__(self, margin=0.01, device="cpu"):
    """
    Initialize the triplet miner.

    Args:
      margin: minimum desired distance between different subjects
    """
    super(OnlineTripletMining, self).__init__()
    self.margin = margin
    self.device = device

  def forward(self, embeddings, labels):
    """
    Compute the loss using the hardest triplets.
    Loss function: TripletMarginWithDistanceLoss (with cosine distance)

    Args:
      embeddings: embeddings to compute loss from
      labels: subject labels
    """

    # Get the pairwise distance matrix, shape (batch_size, batch_size)
    pairwise_dist = 1 - F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

    # Create label masks, shape (batch_size, batch_size)
    indices_equal = torch.eye(labels.size(0)).bool().to(self.device)
    indices_distinct = torch.logical_not(indices_equal)
    labels_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
    label_distinct = torch.logical_not(labels_equal)
    ap_mask = torch.logical_and(indices_distinct, labels_equal)
    an_mask = label_distinct

    # Anchor-positive distances, with invalid ap pairs having a distance of 0
    # Choose hardest positives as max distance for each anchor, shape (batch_size, 1)
    ap_dist = torch.where(ap_mask, pairwise_dist, torch.tensor(0, dtype=pairwise_dist.dtype).to(self.device))
    ap_dist_hard = torch.max(ap_dist, dim=1, keepdim=True)[0]

    # Anchor-negative distances, with invalid an pairs having a distance of 1
    # Choose hardest negatives as min distance for each anchor, shape (batch_size, 1)
    an_dist = torch.where(an_mask, pairwise_dist, torch.tensor(1, dtype=pairwise_dist.dtype).to(self.device))
    an_dist_hard = torch.min(an_dist, dim=1, keepdim=True)[0]

    # Compute triplet margin loss, clamping minimum distance to 0, averaged over the batch
    loss = torch.clamp(ap_dist_hard - an_dist_hard + self.margin, min=0.0).mean()
    return loss


class CatEmbeddingTrainer:
  """
  Helper class for training the cat face embeddings.
  Uses triplet margin loss and an Adam optimizer.
  """

  RETRIEVAL_THRESHOLDS = [1, 5, 10, 25, 50, 100]

  def __init__(self, model, params, train_loader, valid_loader):
    """
    Sets up the classification trainer.

    Args:
      model: model to train
      params: TrainingParams
      train_loader: training data loader
      valid_loader: validation data loader
    """
    self.model = model.to(params.device)
    self.params = params
    self.train_loader = train_loader
    self.valid_loader = valid_loader
    self.logger = Logger(params, train_loader, valid_loader)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)
    self.criterion = OnlineTripletMining(margin=params.loss_margin, device=params.device).to(params.device)
    self.train_embd_db = [[], []]  # database of [embeddings, labels], populated during training and used for evaluation
    self.load_checkpoint()
  
  def save_checkpoint(self, filename="checkpoint.pth"):
    """
    Save checkpoint.
    """
    if self.params.debug:
      print(f"Saving checkpoint to {self.params.save_dir}...")
    torch.save(self.checkpoint, os.path.join(self.params.save_dir, filename))

  def load_checkpoint(self, path=None):
    """
    Load or initialize a checkpoint.

    Args:
      path: if provided, path to checkpoint file to load
    """
    if path:
      self.checkpoint = torch.load(path, map_location=torch.device(self.params.device))
      self.logger.curr_epoch = self.checkpoint["epoch"]
      self.logger.train_counter = self.checkpoint["logged_train_counter"]
      self.logger.valid_counter = self.checkpoint["logged_valid_counter"]
      self.logger.train_losses = self.checkpoint["logged_train_losses"]
      self.logger.valid_metrics = self.checkpoint["logged_valid_metrics"]
      self.model.load_state_dict(self.checkpoint["latest_model_state_dict"])
      self.optimizer.load_state_dict(self.checkpoint["latest_optimizer_state_dict"])
    else:
      self.checkpoint = {
        "epoch": 0,
        "latest_model_state_dict": self.model.state_dict(),
        "latest_optimizer_state_dict": self.optimizer.state_dict(),
        "best_model_state_dict": self.model.state_dict(),
        "best_model_valid_metrics": {f"top-{k}": 0 for k in CatEmbeddingTrainer.RETRIEVAL_THRESHOLDS},
        "logged_train_counter": self.logger.train_counter,
        "logged_valid_counter": self.logger.valid_counter,
        "logged_train_losses": self.logger.train_losses,
        "logged_valid_metrics": self.logger.valid_metrics,
      }

  def train_epoch(self):
    """
    Run one training epoch.
    """
    # Set up
    self.train_embd_db = [[], []]  # [embeddings, labels]
    self.model.train()
    # For each batch, apply backpropagation
    for batch, (X, y) in enumerate(self.train_loader):
      X, y = X.to(self.params.device), y.to(self.params.device)
      # Compute embedding and loss
      embd = self.model(X)
      loss = self.criterion(embd, y)
      # Backpropagation
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      # Log progress
      self.logger.log_train(batch, loss)
      # Store embeddings in db
      self.train_embd_db[0].append(embd)
      self.train_embd_db[1].append(y)
    # Merge lists of tensors into single tensors, ready for evaluation
    self.train_embd_db[0] = torch.cat(self.train_embd_db[0], dim=0)
    self.train_embd_db[1] = torch.cat(self.train_embd_db[1], dim=0)

  def train(self):
    """
    Train and save the model.
    """
    # Initialize progress from checkpoint
    best_model_state = self.checkpoint["best_model_state_dict"]
    best_valid_metrics = self.checkpoint["best_model_valid_metrics"]
    # For each epoch...
    for _ in range(self.params.n_epochs):
      # Train and validate
      self.logger.next_epoch()
      self.train_epoch()
      valid_metrics = self.eval(self.valid_loader)
      self.logger.log_valid(valid_metrics)
      # Save the best-performing model based on retrieval metrics
      for k in CatEmbeddingTrainer.RETRIEVAL_THRESHOLDS:
        if valid_metrics[f"top-{k}"] < best_valid_metrics[f"top-{k}"]:
          break
        elif valid_metrics[f"top-{k}"] > best_valid_metrics[f"top-{k}"]:
          best_valid_metrics = valid_metrics
          best_model_state = self.model.state_dict()
          print("Best model updated.")
          break
      # Update and save checkpoint
      self.checkpoint = {
        "epoch": self.logger.curr_epoch,
        "latest_model_state_dict": self.model.state_dict(),
        "latest_optimizer_state_dict": self.optimizer.state_dict(),
        "best_model_state_dict": best_model_state,
        "best_model_valid_metrics": best_valid_metrics,
        "logged_train_counter": self.logger.train_counter,
        "logged_valid_counter": self.logger.valid_counter,
        "logged_train_losses": self.logger.train_losses,
        "logged_valid_metrics": self.logger.valid_metrics,
      }
      self.save_checkpoint()
      self.logger.plot_performance()
    print("Done!")

  def eval(self, loader):
    """
    Evaluate model on some data on retrieval metrics
    using nearest-neighbor classification.

    Args:
      loader: data to test on
    
    Returns:
      Retrieval metrics
    """
    # Set up
    eval_embd = [[], []]  # [embeddings, labels]
    eval_train_classmap = { eval_id: self.train_loader.classes.index(eval_label) for eval_id, eval_label in enumerate(loader.classes) }  # eval class id --> train class id
    self.model.eval()
    # Run eval
    with torch.no_grad():
      for X, y in loader:
        X, y = X.to(self.params.device), y.to(self.params.device)
        # Compute embedding
        embd = self.model(X)
        # Record embeddings
        eval_embd[0].append(embd)
        eval_embd[1].append(y)

    # Merge lists of tensors into single tensors, ready for evaluation
    eval_embd[0] = torch.cat(eval_embd[0], dim=0)
    eval_embd[1] = torch.cat(eval_embd[1], dim=0)

    # Compute pairwise cosine similiary distances between train db and eval embeddings, shape (train_size, eval_size)
    pairwise_dist = 1 - F.cosine_similarity(self.train_embd_db[0].unsqueeze(1), eval_embd[0].unsqueeze(0), dim=2)

    # Get indices of top 100 nearest neighbors, shape (eval_size, 100)
    _, indices = torch.topk(pairwise_dist, k=100, dim=1)
    indices = indices.to(self.params.device)

    # Evaluate retrieval
    metrics = {f"top-{k}": 0 for k in CatEmbeddingTrainer.RETRIEVAL_THRESHOLDS}
    eval_size = eval_embd[0].size(0)
    # For each embedding to evaluate, accumulate the corresponding metrics
    for i in range(eval_size):
      # ground truth label, number
      y = eval_embd[1][i].item()
      # Convert eval label to train label, since class ids may not be aligned
      y = eval_train_classmap[y]
      # top 100 predictions
      top_100 = self.train_embd_db[1][indices[i]]
      # Update each metric
      for k in CatEmbeddingTrainer.RETRIEVAL_THRESHOLDS:
        if y in top_100[:k]:
          metrics[f"top-{k}"] += 1
    
    # Average metrics over eval_size
    metrics = { m: v/eval_size if v > 0 else 0 for m, v in metrics.items() }
    return metrics




####################
# MAIN
####################

def dir_file_pairs(arg):
  """
  Custom argparse type.
  Takes in a list of directory/file pairs formatted: pair[::pair[::pair[...]]]
  Each file in the pair is separated by a comma: dir,file

  Example:
    ./data/train,./data/labels.csv
  """
  try:
    # Split the input argument by colons to get pairs
    pairs = arg.split("::")
    # Split each pair by comma
    pairs = [tuple(pair.split(",")) for pair in pairs]
    return pairs
  except ValueError:
    raise argparse.ArgumentTypeError("Invalid pair format. Should be dir1,file1::dir2,file2::...")

def main(argv):
  """
  Main function.
  Loads cat face data, then trains a deep network on cat face embeddings.
  The trained model and its training performance graph are saved.
  """
  
  # handle any command line arguments
  parser = argparse.ArgumentParser()
  # (training params)
  parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
  parser.add_argument("--debug", dest="debug", action="store_true", help="Show extra debug output")
  parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
  parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
  parser.add_argument("--loss_margin", type=float, default=0.01, help="Margin for loss function")
  parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
  parser.add_argument("--checkpoint_path", type=str, default=None, help="Checkpoint path to load before training")
  parser.add_argument("--save_dir", type=str, default="./out", help="Where to save output")
  parser.add_argument("--logging_freq", type=int, default=1, help="How often to log training progress")
  # (data params)
  parser.add_argument("--augmented_data", type=dir_file_pairs, default=[], help="List of data augmentation directory/file pairs formatted dir1,file1::dir2,file2::...")
  parser.add_argument("train_dir", type=str, help="Directory with training data")
  parser.add_argument("valid_dir", type=str, help="Directory with validation data")
  parser.add_argument("test_dir", type=str, help="Directory with test data")
  parser.add_argument("labels_path", type=str, help="Path to annotation file")


  # parse parameters
  args = parser.parse_args()
  hp = TrainingParams(
      seed = args.seed,
      batch_size = args.batch_size,
      lr = args.lr,
      n_epochs = args.n_epochs,
      checkpoint_path = args.checkpoint_path,
      save_dir = args.save_dir,
      logging_freq = args.logging_freq,
      debug = args.debug
  )
  dp = DataParams(
    test_dir=args.test_dir,
    train_dir=args.train_dir,
    valid_dir=args.valid_dir,
    labels_path=args.labels_path,
    augmented_data=args.augmented_data
  )

  # make code repeatable
  torch.manual_seed(hp.seed)
  torch.backends.cudnn.enabled = False 
  
  print("Using training parameters:")
  print(vars(hp))
  print()

  # load data (only batch the training set)
  print("-------------------------  LOADING CAT FACE DATA...  -------------------------")
  train_loader = CatFaceDataLoader(dp.train_files, batch_size=hp.batch_size)
  valid_loader = CatFaceDataLoader(dp.valid_files, batch_size=hp.batch_size)
  test_loader = CatFaceDataLoader(dp.test_files, batch_size=hp.batch_size)
  print("-------------------------  CAT FACE DATA LOADED!     -------------------------\n")

  # prepare model
  print("-------------------------  INITIALIZING MODEL...     -------------------------")
  model = CatEmbedNN().to(hp.device)
  print("-------------------------  MODEL READY!              -------------------------\n")

  # train
  print("-------------------------  TRAINING BEGIN...         -------------------------")
  trainer = CatEmbeddingTrainer(model, hp, train_loader, valid_loader)
  trainer.train()
  print("-------------------------  TRAINING COMPLETE!        -------------------------\n")

  # evaluate on test data
  print("-------------------------  EVALUATING ON TEST SET... -------------------------")
  test_metrics = trainer.eval(test_loader)
  for m, val in test_metrics.items():
    print(f"|  {m}: {val}")
  print("-------------------------  TEST COMPLETE!            -------------------------\n")





if __name__ == "__main__":
  main(sys.argv)