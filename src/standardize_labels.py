"""
Author: Elizabeth Witten

Standardizes face labeling from different datasets by extracting a face bounding box
and outputing in CSV format: filename, xmin, ymin, xmax, ymax
"""

###############
# IMPORTS
###############

import sys
import argparse
import os
import xml.etree.ElementTree as ET
import json
import pandas as pd




###############
# FUNCTIONS
###############

def process_msft(label_dir, out_path, append=True):
  """
  Process and convert MSFT cat dataset landmark points to bounding box.

  Args:
    label_dir: directory with .cat files
    out_path: path to output csv file
    append: whether to append or overwrite output csv file
  """

  # Store data for each cat file
  data = []
  all_cat = [ fn for fn in os.listdir(label_dir) if fn.endswith(".cat")]
  n_files = len(all_cat)
  for i, filename in enumerate(all_cat):
     
    # Read the text file
    full_filename = os.path.join(label_dir, filename)
    with open(full_filename, "r") as file:
       landmarks = [ int(pt) for pt in file.read().split() ]
    
    # parse the landmark points
    l_eye   = (max(0, landmarks[1]),  max(0, landmarks[2]))
    r_eye   = (max(0, landmarks[3]),  max(0, landmarks[4]))
    mouth   = (max(0, landmarks[5]),  max(0, landmarks[6]))
    l_ear_o = (max(0, landmarks[7]),  max(0, landmarks[8]))
    l_ear_t = (max(0, landmarks[9]),  max(0, landmarks[10]))
    l_ear_i = (max(0, landmarks[11]), max(0, landmarks[12]))
    r_ear_i = (max(0, landmarks[13]), max(0, landmarks[14]))
    r_ear_t = (max(0, landmarks[15]), max(0, landmarks[16]))
    r_ear_o = (max(0, landmarks[17]), max(0, landmarks[18]))

    # first capture the face as a square
    # use max width, excluding ear tips
    width = max([
        abs(l_eye[0] - r_eye[0]),
        abs(l_ear_o[0] - r_ear_o[0]),
        abs(l_ear_i[0] - r_ear_i[0]),
    ])
    height = width
    # centered between the eyes
    center = ((l_eye[0] + r_eye[0]) // 2,
              (l_eye[1] + r_eye[1]) // 2)
    # initial bounding box
    xmin = center[0] - width//2
    ymin = center[1] - height//2
    if xmin < 0:                  # oob width adjustments
      width += xmin
      xmin = 0
    if ymin < 0:                  # oob height adjustments
      height += ymin
      ymin = 0
    xmax = xmin + width
    ymax = ymin + height

    # now adjust the bounding box to fit the ear tips
    # adjust top of box
    ymin = min(ymin, l_ear_t[1], r_ear_t[1])
    # adjust left of box
    xmin = min(xmin, l_ear_t[0], r_ear_t[0])
    # adjust right of box
    xmax = max(xmax, l_ear_t[0], r_ear_t[0])

    # get associated image filename by stripping off the .cat extension
    img_filename = filename[:-4]

    # Store data as dictionary
    data.append({
      "filename": img_filename,
      "xmin": xmin,
      "ymin": ymin,
      "xmax": xmax,
      "ymax": ymax
    })
    

    # Progress output
    if (i+1) % 50 == 0:
      print(f"Processed {i+1} files...")
  
  # Convert data to dataframe
  df = pd.DataFrame(data)

  # Save csv
  if append:
    df.to_csv(out_path, mode='a', index=False, header=not os.path.exists(out_path))
  else:
    df.to_csv(out_path, mode='w', index=False)
  
  print(f"Labels saved to {out_path}!\n")


def process_oxiiit(label_dir, out_path, append=True):
  """
  Process and convert Oxford cat dataset XML bounding box.

  Args:
    label_dir: directory with .xml files
    out_path: path to output csv file
    append: whether to append or overwrite output csv file
  """

  # Store data for each XML file
  data = []
  all_xml = [ fn for fn in os.listdir(label_dir) if fn.endswith(".xml")]
  n_files = len(all_xml)
  for i, filename in enumerate(all_xml):

    # Read the XML from file
    full_filename = os.path.join(label_dir, filename)
    with open(full_filename, "r") as file:
      xml_str = file.read()

    # Parse the XML string and extract bounding box values
    root = ET.fromstring(xml_str)
    img_filename = root.find("filename").text
    xmin = root.find(".//xmin").text
    ymin = root.find(".//ymin").text
    xmax = root.find(".//xmax").text
    ymax = root.find(".//ymax").text

    # Store data as dictionary
    data.append({
      "filename": img_filename,
      "xmin": xmin,
      "ymin": ymin,
      "xmax": xmax,
      "ymax": ymax
    })

    # Progress output
    if (i+1) % 50 == 0:
      print(f"Processed {i+1} files...")
  
  # Convert data to dataframe
  df = pd.DataFrame(data)

  # Save csv
  if append:
    df.to_csv(out_path, mode='a', index=False, header=not os.path.exists(out_path))
  else:
    df.to_csv(out_path, mode='w', index=False)
  
  print(f"Labels saved to {out_path}!\n")


def process_labelstudio(label_path, out_path, append=True):
  """
  Process and convert cat faces labeled in LabelStudio.

  Args:
    label_dir: directory with .cat files
    out_path: path to output csv file
    append: whether to append or overwrite output csv file
  """

  # Read in json file
  with open(label_path) as json_file:
    raw_data = json.load(json_file)

  # JSON normalize to dataframe
  df = pd.json_normalize(raw_data, "label", ["image"]).drop(["rotation"], axis=1)

  # Extract original file name
  df[["temp", "filename"]] =  df["image"].str.split("-", n=1, expand=True)
  df =  df.drop(["image", "temp"], axis=1)

  # Drop label
  df =  df.drop(["rectanglelabels"], axis=1)

  # Extract corners
  df["xmin"] = (df["original_height"] * (df["y"] / 100)).astype(int)
  df["ymin"] = (df["original_width"]  * (df["x"] / 100)).astype(int)
  df["xmax"] = (df["original_height"] * ((df["y"] +  df["height"]) / 100)).astype(int)
  df["ymax"] = (df["original_width"]  * ((df["x"] +  df["width"]) / 100)).astype(int)
  df = df.drop(["original_height", "original_width", "height", "width", "x", "y"], axis=1)

  # Save csv
  if append:
    df.to_csv(out_path, mode='a', index=False, header=not os.path.exists(out_path))
  else:
    df.to_csv(out_path, mode='w', index=False)

  print(f"Labels saved to {out_path}!\n")




###############
# MAIN
###############

def main(argv):
  """
  Main function.
  Parse annotation sets and append into one CSV file.
  """
  
  # handle any command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("--oxford_dir", type=str, default=None)
  parser.add_argument("--msft_dir", type=str, default=None)
  parser.add_argument("--labelstudio_dir", type=str, default=None)
  parser.add_argument("--out_path", type=str, default="./labels.csv")
  args = parser.parse_args()
  
  if args.msft_dir:
    process_msft(msft_dir, args.out_path, append=True)
  
  if args.oxford_dir:
    process_oxiiit(oxford_dir, args.out_path, append=True)

  if args.labelstudio_dir:
    process_labelstudio(args.labelstudio_dir, args.out_path, append=True)


if __name__ == "__main__":
  main(sys.argv)