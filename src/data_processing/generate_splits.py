"""
Author: Elizabeth Witten

Utilities for splitting data into test/train/validation sets.
"""

###############
# IMPORTS
###############

import sys
import argparse
import os
import shutil




###############
# FUNCTIONS
###############


def make_split_dir(base_dir, split, id):
  """
  Make the specified dataset split directory.

  Args:
    base_dir: directory containing the split directories
    split: one of "test", "train", "valid"
    id: subject id, name of sub-directory
  
  Returns: 
    path of created directory
  """
  split_dir = os.path.join(base_dir, split, id)
  if not os.path.exists(split_dir):
    os.makedirs(split_dir)
  return split_dir


def split_dataset(src_dir, out_base_dir):
  """
  Split the images contained under one directory into three new directories:
  train, test, valid. These are copies, the source directory remains unchanged.

  Args:
    src_dir: directory containing the data to split
    out_base_dir: base directory to output the new split directories
  """
  
  for id in os.listdir(src_dir):

    # each directory is the subject id
    src_subdir = os.path.join(src_dir, id)
    if not os.path.isdir(src_subdir):
      continue
    
    # get list of subject images
    subj_imgs = os.listdir(src_subdir)
    
    # distribute images into splits, at least 2 training images, followed by test and valid
    test_imgs = []
    train_imgs = []
    valid_imgs = []
    for i, img in enumerate(subj_imgs):
      if i == 0 or i % 3 == 1:
        train_imgs.append(img)
      elif i % 3 == 2:
        test_imgs.append(img)
      else:
        valid_imgs.append(img)
    
    # copy the distributed images
    if test_imgs:
      test_dir = make_split_dir(out_base_dir, "test", id)
      for img in test_imgs:
        shutil.copyfile(os.path.join(src_subdir, img), os.path.join(test_dir, img))
    if train_imgs:
      train_dir = make_split_dir(out_base_dir, "train", id)
      for img in train_imgs:
        shutil.copyfile(os.path.join(src_subdir, img), os.path.join(train_dir, img))
    if valid_imgs:
      valid_dir = make_split_dir(out_base_dir, "valid", id)
      for img in valid_imgs:
        shutil.copyfile(os.path.join(src_subdir, img), os.path.join(valid_dir, img))




###############
# MAIN
###############

def main(argv):
  """
  Main function.
  Generate data splits.
  """
  
  # handle any command line arguments
  # example: generate_splits.py "./data/by_id" "./data/split"
  parser = argparse.ArgumentParser()
  parser.add_argument("src_dir", type=str)
  parser.add_argument("out_dir", type=str)
  args = parser.parse_args()

  split_dataset(args.src_dir, args.out_dir)


if __name__ == "__main__":
  main(sys.argv)
