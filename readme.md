# Learned Cat Face Embeddings

**Author: Elizabeth Witten**

**Project Description**:
This project implements a deep neural network for learning cat face embeddings, utilizing transfer learning from the pre-trained GoogLeNet model. The proposed model is trained using triplet margin loss to learn a 256-D embedding space where images of the same subject’s face are located close together. Finally, the resulting embedding space is analyzed from two perspectives to identify potential areas for improvement for the task of cat identification. First, the entire space is visualized using the t-SNE algorithm to project it into two dimensions. Second, the trained model is used to create a labeled database of cat face embeddings. A command-line interface (CLI) application is developed to interface with the database, allowing for the addition of new images and performing top-k retrieval for query images of cats.

**Project Presentation**: https://youtu.be/PKsguz6dteM

---

## Repository Organization
- **db** contains the database file(s)
- **scripts** contains shell scripts for running the embedding and identification programs
  - *Note: shell scripts are currently written relative to the project root, not the scripts folder*
- **src** contains the source code for this project
  - **src/data_processing** contains source code related to generating the train/valid/test sets
  - **src/embeddings** contains source code for the learned embedding model and embedding database generation
- **trained_model** contains the trained model checkpoint and supporting documents

## How to run

> Tip: Running any python script with the -h flag will output a descriptive usage message.

### Setup
- Clone this repository.
- In the repository directory, create and activate a virtual environment for this project.
- Install required packages listed in `requirements.txt`

### Dataset Generation

> Due to size contraints, the dataset of images used to train the provided model is not made available on GitHub, and instead can be shared upon request.

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
  |----- [subject folders]
  |--- train
  |----- [subject folders]
  |--- valid
  |----- [subject folders]
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
