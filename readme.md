# Learned Cat Face Embeddings

**Author: Elizabeth Witten**

This project implements a learned cat face embedding model, with an interactive CLI application to identify cat faces in an image.

---

## Repository Organization
- **src** contains the source code for this project
  - **src/data_processing** contains source code related to generating the train/valid/test sets
  - **src/embeddings** contains source code for the learned embedding model and embedding database generation

## Setup
1. Clone this repository.
2. In the repository directory, create and activate a virtual environment for this project.

## How to run

### Dataset Generation
- The image data should be organized to be parsed into a `torchvision.datasets.ImageFolder`.
  - The root data directory should contain subfolders.
  - Each subfolder should only contain images with one cat subject, and the subfolder should be named with the subject id assigned to the cat pictured.
- Run `src/data_processing/standardize_labels.py` to process the image annotations into a common format.
  - If using multiple sources (and therefore have multiple annotation files) to compile the dataset, this script will also compile the labels into one CSV file.
- Run `src/data_processing/generate_splits.py` to split the root directory into train/valid/test sets.
- If desired, run `src/data_processing/augment_data.py` to generate augmented training data.
- The structure of the data folder should now look something like:
  ```
  data_root
  |- orig_split
  |--- test
  |--- train
  |--- valid
  |--- orig_labels.csv
  |- augmented
  |--- [subject folders]
  |--- augmented_labels.csv
  ```

### Training the Embedding Model
- Run `src/embeddings/train_embeddings.py` to train the embedding model.
- Run `src/embeddings/generate_database.py` to generate a database of labeled embeddings using the trained model.

### Visualizing the Learned Embeddings
- Run `src/embeddings/visualize_tsne.py` to visualize a 2D representation of the learned embedding space.

### Cat Identification
- Run `src/identify_cats.py` to enter the interactive cat identification application.
