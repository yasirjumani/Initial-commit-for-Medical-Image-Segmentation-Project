Semi-Supervised Deep Learning for Skin Lesion Segmentation: An Annotation Ambiguity Aware Approach
This repository contains the code and report for a deep learning project focused on automated skin lesion segmentation from dermoscopic images. The project utilizes a novel MultiDecoderCNN architecture within a semi-supervised learning framework to not only segment lesions but also to quantify and highlight areas of annotation ambiguity through uncertainty maps.

Table of Contents
Introduction

Key Features

Technical Report

Dataset

Project Structure

Setup and Installation

Usage

Results

Contributing

License

Contact

Introduction
Accurate segmentation of skin lesions is crucial for the early detection and diagnosis of melanoma. This project addresses the challenges of limited labeled medical data and inherent annotation subjectivity by developing a semi-supervised deep learning model. Our MultiDecoderCNN architecture learns to segment lesions and provides valuable pixel-wise uncertainty information, assisting dermatologists by indicating regions where the model is less confident due to potential annotation ambiguities or complex lesion characteristics.

Key Features
MultiDecoderCNN Architecture: A custom Fully Convolutional Network (FCN) based architecture with two decoders for robust feature learning and uncertainty estimation.

Semi-Supervised Learning: Employs consistency regularization to effectively leverage both labeled and unlabeled data, improving generalization with limited annotations.

Annotation Ambiguity Awareness: Generates pixel-wise uncertainty maps by analyzing the divergence between the two decoders, providing crucial insights into challenging segmentation regions.

Comprehensive Data Augmentation: Robust augmentation pipeline to enhance model resilience to variations in dermoscopic image appearance.

Combined Loss Function: Utilizes a combination of Binary Cross-Entropy (BCE) and Dice Loss for effective training on imbalanced medical datasets.

Technical Report
For a detailed explanation of the project's background, methodology, experimental setup, and comprehensive results, please refer to the technical report available in this repository:

report.pdf (or report.tex if you're compiling it)

Dataset
This project utilizes the ISIC 2018 Challenge: Lesion Boundary Segmentation (Task 1) dataset. Due to its large size (~15 GB) and GitHub's file size limitations, the dataset is not included in this repository.

You can download the dataset directly from the official ISIC archive:

ISIC 2018 Dataset Download Page: https://challenge.isic-archive.com/data/#2018

After downloading, please structure your dataset as follows relative to the project root:

medical_image_segmentation/
├── dataset/
│   ├── ISIC2018_Task1-2_Training_Input/
│   │   ├── ISIC_XXXXXXXX.jpg
│   │   └── ...
│   ├── ISIC2018_Task1_Training_GroundTruth/
│   │   ├── ISIC_XXXXXXXX_segmentation.png
│   │   └── ...
│   ├── ISIC2018_Task1-2_Validation_Input/
│   │   ├── ISIC_XXXXXXXX.jpg
│   │   └── ...
│   ├── ISIC2018_Task1_Validation_GroundTruth/
│   │   ├── ISIC_XXXXXXXX_segmentation.png
│   │   └── ...
│   ├── ISIC2018_Task1-2_Test_Input/
│   │   ├── ISIC_XXXXXXXX.jpg
│   │   └── ...
│   └── ISIC2018_Task1_Test_GroundTruth/ (Optional, for final evaluation after training)
│       ├── ISIC_XXXXXXXX_segmentation.png
│       └── ...
├── train_model.py
├── evaluate_model.py
├── analyze_predictions.py
├── plot_losses.py
├── report.tex
├── report.pdf (if compiled)
├── requirements.txt
├── .gitignore
└── ... (other project files)

Note: The train_model.py and evaluate_model.py scripts assume these specific relative paths or require updating their image_dir and mask_dir variables if you place the dataset elsewhere.

Project Structure
.
├── dataset/                  # (NOT in Git) Downloaded ISIC 2018 dataset
├── output/                   # (Ignored by Git) Saved model checkpoints, predictions
├── visualizations/           # Example segmentation and uncertainty maps
│   ├── ISIC_0012169_segmentation_and_uncertainty.png
│   └── ...
├── src/                      # (Optional: for larger projects, could put .py files here)
│   ├── models/
│   └── data/
├── train_model.py            # Main script for training the MultiDecoderCNN
├── evaluate_model.py         # Script for evaluating model performance
├── analyze_predictions.py    # Script to generate uncertainty maps and visual analysis
├── plot_losses.py            # Script to plot training/validation loss curves from logs
├── requirements.txt          # Python dependencies
├── report.tex                # LaTeX source for the technical report
├── report.pdf                # Compiled PDF version of the technical report
├── README.md                 # This file
└── .gitignore                # Specifies files/folders to ignore in Git

Setup and Installation
Clone the repository:

git clone https://github.com/yasirjumani/Initial-commit-for-Medical-Image-Segmentation-Project.git
cd Initial-commit-for-Medical-Image-Segmentation-Project

Create a Python virtual environment:

python3 -m venv venv

Activate the virtual environment:

On macOS/Linux:

source venv/bin/activate

On Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

(Ensure your requirements.txt lists all necessary packages like torch, torchvision, numpy, matplotlib, Pillow, etc.)

Download the Dataset: As mentioned in the Dataset section, download the ISIC 2018 dataset and place it in the dataset/ directory at the root of this project.

Usage
Training the Model
To train the MultiDecoderCNN model with semi-supervised learning and consistency regularization:

python train_model.py

This script will save model checkpoints to the output/ directory.

Evaluating the Model
To evaluate the performance of a trained model:

python evaluate_model.py

This script will calculate Dice and IoU scores on the validation/test set.

Generating Uncertainty Maps
To visualize segmentations and corresponding uncertainty maps (as shown in the report's visualizations/ examples):

python analyze_predictions.py

Plotting Loss Curves
To generate the loss_curves.png plot from your training logs:

python plot_losses.py

This will create loss_curves.png in the project root.

Results
Quantitative and qualitative results, including Dice and IoU scores for the baseline and semi-supervised models, are thoroughly discussed in the Technical Report. The report also includes examples of generated segmentation masks and uncertainty maps.

Contributing
Feel free to open issues or submit pull requests if you have suggestions for improvements or bug fixes.

License
[Specify your license here, e.g., MIT License, Apache 2.0, etc.]

Contact
For any questions or inquiries, please contact:

Yasir Ahmed

Email: yasir.ahmed@example.com (Replace with your actual email)

GitHub: github.com/yasirjumani
