# UATR: Computation Generalized Embeddings with Unsupervised Contrastive Learning

## Overview

**UATR** is a Python-based framework designed to compute generalized embeddings using unsupervised contrastive learning (CL). This repository provides tools and utilities to facilitate the training and evaluation of models leveraging CL techniques for various applications.

## Features

- **Encoder Implementation**: &#x25A0; A customizable encoder architecture to process input data and generate embeddings.
- **Loss Functions**: &#x25A0; Implementations of contrastive loss functions to optimize the embeddings.
- **Data Augmentation**: &#x25A0; Functions to augment training data, enhancing model robustness.
- **Efficient Data Loading**: &#x25A0; Optimized data loading mechanisms to handle large datasets effectively.
- **Training Scripts**: &#x25A0; Scripts to facilitate model training, including support for job scheduling.
- **LARS Optimizer**: &#x25A0; Implementation of the LARS optimizer to improve training speed and performance.
  
## Repository Structure

The repository contains the following key files and directories:

- `Encoder.py`: &#x25A0; Defines the encoder architecture for embedding generation.
- `loss_functions.py`: &#x25A0; Contains implementations of contrastive loss functions.
- `Augmentation_functions_torch.py`: &#x25A0; Provides data augmentation utilities.
- `efficient_data_loader.py`: &#x25A0; Includes optimized data loading functions.
- `Training.py`: &#x25A0; Script to train the model using the defined components.
- `job_script.sh`: &#x25A0; Shell script for job scheduling and execution.
- `lars.py`: &#x25A0; Implementation of the LARS optimizer.

## Installation

To use UATR, clone the repository and install the required dependencies:

```bash
git clone https://github.com/hildeingvildhummel/UATR.git
cd UATR
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model, run the `job_script.sh` bash script and modify the command-line arguments to adjust parameters such as learning rate, batch size, and number of epochs.


## Related Work

For further context on this approach, refer to the preprint: The Computation of Generalized Embeddings for Underwater Acoustic Target Recognition Using Contrastive Learning (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5112948) for more details on the theoretical foundations and experimental results.



