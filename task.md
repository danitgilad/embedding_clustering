# Senior  - Take-Home Assignment

## Overview

This assignment consists of two independent unsupervised learning tasks over different data modalities.  
Your goal is to demonstrate your ability to research unfamiliar domains, make sound technical decisions, and deliver production-quality code.

**Important:** The solution must be delivered as a **well-structured Python project** - not as Jupyter notebooks or flat scripts.


---

## Project Structure Requirements

Your deliverable will be evaluated on code quality and project organization as much as on the analysis itself. The project must include:

- A well-organized module structure with meaningful separation of concerns (e.g., a `src/` directory with separate modules using `__init__.py` files). Note: this does **not** need to be an installable Python package.
- A clear entry point (e.g., `main.py` or a CLI using `argparse` ) that allows running each part of the assignment.
- Configuration separated from logic - no hardcoded paths scattered throughout the code.
- A `requirements.txt` listing all dependencies.
- A `README.md` containing:
  - Setup and installation instructions.
  - How to run each part of the assignment.
  - A summary of your approach, findings, and key decisions for each part.
- Proper use of logging (avoid bare `print()` statements for operational output).
- Meaningful function, class, and module names.
- Docstrings and type hints are encouraged.

---

## Part A - Unsupervised Clustering of 3D Assets


### Objective

Analyze and cluster 3D assets of glasses based on their geometric and visual features, such that each cluster groups glasses with a similar appearance.
Note that there is no single correct answer — this is an inherently subjective task.


### Instructions

1. **Dataset Exploration**
   - Use the provided `.glb` files (located in the `assets/` folder) as your dataset of 3D assets.
   - Load and explore the assets using a suitable library (e.g., `trimesh`, `Open3D`, `pygltflib`, `pytorch3d`).
   - Examine the internal structure of each file, paying attention to how mesh components and materials are organized.

2. **Feature Extraction**
   - Research and select appropriate methods for extracting meaningful features from the 3D assets.
   - Extract **at least two types of features**:
     - One **2D visual feature** derived from rendered images of the assets.
     - One **3D geometric feature** computed directly from the mesh, without relying on rendering.
   - Analyze the extracted features, exploring their distributions and discriminative properties.

3. **Clustering Analysis**
   - Apply clustering algorithms or other unsupervised learning techniques to group assets based on the extracted features.
   - Compare the expressiveness of different feature types and evaluate how each affects the resulting clusters.

4. **Visualization & Reporting**
   - Visualize the clusters and provide a summary of the similarities found between the assets.
   - Document the process and findings - including challenges and observations - in your `README.md`.

---

## Part B - Unsupervised Classification of Face Images Using Pretrained Embeddings

### Objective

Classify and analyze features in facial images using representative embeddings extracted from a pretrained model, applied to a custom-generated dataset.

### Instructions

1. **Dataset Creation**
   - Use [thispersondoesnotexist.com](https://thispersondoesnotexist.com/) (or an equivalent AI face generator) to programmatically generate and save a dataset of facial images locally.
   - Document your generation approach, dataset size, and any preprocessing applied.

2. **Pretrained Model Selection**
   - Research and select a suitable pretrained model (classification, detection, or segmentation).
   - Justify your choice in the `README.md`.

3. **Embedding Extraction**
   - Feed the dataset into the pretrained model and extract representative feature embeddings for each image.

4. **Feature Analysis & Classification**
   - Analyze the extracted feature embeddings and explore their properties and patterns.
   - Apply clustering algorithms or other unsupervised learning techniques to classify the images based on their representative vectors.
   - Evaluate and iterate on your approach to improve classification quality.

---

## Evaluation Criteria

Your submission will be evaluated on the following dimensions:

| Criterion | What we look for |
|---|---|
| **Code quality** | Modularity, readability, separation of concerns, DRY principles |
| **Project structure** | Proper project organization, clear entry points, configuration management |
| **Engineering practices** | Logging, error handling, type hints, docstrings |
| **Research ability** | Thoughtful selection of tools, models, and techniques with clear justification |
| **Analysis depth** | Quality of feature extraction, clustering methodology, and evaluation |
| **Visualization** | Clear, informative visualizations that support the analysis |
| **Documentation** | Well-written README covering setup, usage, approach, and findings |

---

## Deliverables

Submit a **zip archive or git repository** containing:

- A structured Python project (not notebooks or flat scripts).
- `README.md` with setup instructions, how to run, and a summary of approach and findings.
- `requirements.txt` for reproducible setup.
- Output visualizations (saved as image files).
- The `assets/` folder should **not** be included in the submission - we already have the data.
