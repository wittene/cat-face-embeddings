"""
Author: Elizabeth Witten

Utilities for image augmentation.
"""

###############
# IMPORTS
###############

import sys
import argparse
import cv2
import numpy as np
import pandas as pd
import os
import random




###############
# CONSTANTS
###############

# List of augmentation techniques to apply
APPLY_AUGS = ["rotation", "lighting", "erasing"]

# Parameters for data augmentation
ROTATION_RANGE = [-15, 15]   # Rotation angles in degrees
LIGHTING_RANGE = [0.8, 1.2]  # Lighting factors for brightness and contrast
ERASE_RANGE    = [0.1, 0.3]  # Ratios of the bounding box area to be erased




###############
# FUNCTIONS
###############

def augment_images(img_dir, label_path, out_dir, out_subdir = "", n=3):
  """
  Recursively augment all .jpg images in given directory and sub-directories.
  
  Augmented images are saved to the output directory with added suffix:
  -->  [original_file_name.jpg].[0-(n-1)].augmented.jpg
  
  The labels are also updated as needed (i.e. after rotation), and saved to the 
  output directory.

  Args:
    img_dir: directory containing images or sub-directories of images
    label_path: path to csv labels file
    n: how many augmented images to produce per sample
    out_dir: directory to contain augmented images, will mirror the structure of img_dir
  """

  print(f"Processing {os.path.join(out_dir, out_subdir)}...")

  # Load original labels
  df_orig = pd.read_csv(label_path)

  # Initialize augmented labels
  df_aug = pd.DataFrame(columns=df_orig.columns)

  for filename in os.listdir(img_dir):

    file_path = os.path.join(img_dir, filename)

    if os.path.isdir(file_path) and not filename.startswith("."):
      # Recursively apply data augmentation to subdirectories
      augment_images(file_path, label_path, out_dir, out_subdir=os.path.join(out_subdir, filename), n=n)

    elif filename.endswith(".jpg"):

      # Perform and save n augmentations
      for i in range(n):

        # Load the input image
        img = cv2.imread(file_path)

        # Make a copy of the original label
        label = df_orig[df_orig.filename == filename].iloc[0].copy()
        
        # Apply data augmentation techniques
        for technique in APPLY_AUGS:

          if technique == "rotation":
            # Random rotation
            angle = random.uniform(ROTATION_RANGE[0], ROTATION_RANGE[1])
            rows, cols = img.shape[:2]
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            # Apply transform to image
            img = cv2.warpAffine(img, M, (cols, rows))
            # Apply transform to bounding box
            center = (int(label.xmin + label.xmax) // 2, int(label.ymin + label.ymax) // 2)
            size = (int(label.xmax - label.xmin), int(label.ymax - label.ymin))
            rect_pts = np.array([cv2.boxPoints((center, size, 0))])
            rect_pts = cv2.transform(rect_pts, M)
            # Update the label
            (label.xmin, label.ymin) = tuple(rect_pts[0].min(0))
            (label.xmax, label.ymax) = tuple(rect_pts[0].max(0))
            label.xmin = max(0, label.xmin)
            label.ymin = max(0, label.ymin)

          elif technique == "lighting":
            # Random brightness and contrast
            alpha = random.uniform(LIGHTING_RANGE[0], LIGHTING_RANGE[1])
            beta = random.uniform(LIGHTING_RANGE[0], LIGHTING_RANGE[1])
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

          elif technique == "erasing":
            # Random erasing of rectangular patches within a bounding box
            bbox = img[ int(label.ymin):int(label.ymax), int(label.xmin):int(label.xmax) ]
            rows, cols = bbox.shape[:2]
            x = random.randint(0, cols)
            y = random.randint(0, rows)
            width = random.randint(int(ERASE_RANGE[0] * cols), int(ERASE_RANGE[1] * cols))
            height = random.randint(int(ERASE_RANGE[0] * rows), int(ERASE_RANGE[1] * rows))
            bbox[y:y+height, x:x+width, :] = 0

        # Save the augmented image
        output_filename = f"{filename}.{i}.augmented.jpg"
        output_file_path = os.path.join(out_dir, out_subdir, output_filename)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        cv2.imwrite(output_file_path, img)

        # Save augmented image labels
        label.filename = output_filename
        df_aug = pd.concat([df_aug, label.to_frame().T], ignore_index=True)
      
  # Save the augmented labels file, appending if file exists
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  csv_path = os.path.join(out_dir, os.path.basename(label_path))
  df_aug.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))




###############
# MAIN
###############

def main(argv):
  """
  Main function.
  Generate data augmentation.
  """
  
  # handle any command line arguments
  # example: augment_images.py --n=3 "./data/by_id/" "./data/labels.csv" "./data/by_id_augmented/"
  parser = argparse.ArgumentParser()
  parser.add_argument("img_dir", type=str)
  parser.add_argument("label_path", type=str)
  parser.add_argument("out_dir", type=str)
  parser.add_argument("--n", type=int, default=3)
  args = parser.parse_args()

  augment_images(args.img_dir, args.label_path, args.out_dir, n=args.n)


if __name__ == "__main__":
  main(sys.argv)
