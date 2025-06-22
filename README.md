# Semi-Supervised Deep Learning for Skin Lesion Segmentation: An Annotation Ambiguity Aware Approach

This repository contains the code and report for a deep learning project focused on automated skin lesion segmentation from dermoscopic images. The project utilizes a novel MultiDecoderCNN architecture within a semi-supervised learning framework to segment lesions and quantify annotation ambiguity through uncertainty maps.

## Table of Contents

- Introduction
- Key Features
- Technical Report
- Dataset
- Project Structure
- Setup and Installation
- Usage
- Results
- Contributing
- License
- Contact

## Introduction

Accurate segmentation of skin lesions is crucial for early detection and diagnosis of melanoma. This project addresses the challenges of limited labeled data and annotation subjectivity by developing a semi-supervised model. The MultiDecoderCNN architecture learns to segment lesions and provides pixel-wise uncertainty maps, highlighting areas of potential ambiguity.

## Key Features

- **MultiDecoderCNN Architecture**: Custom FCN with dual decoders for segmentation and uncertainty estimation.
- **Semi-Supervised Learning**: Consistency regularization to utilize both labeled and unlabeled data.
- **Annotation Ambiguity Awareness**: Uncertainty maps derived from decoder divergence.
- **Robust Data Augmentation**: Enhances generalization to varied dermoscopic images.
- **Combined Loss Function**: Binary Cross-Entropy and Dice Loss for improved training on imbalanced data.

## Technical Report

[Download the technical report (PDF)](./Semi_Supervised_Deep_Learning_for_Skin_Lesion_Segmentation.pdf)

## Dataset

ISIC 2018 Challenge: Lesion Boundary Segmentation (Task 1)

Dataset not included in this repository due to size. Download from: https://challenge.isic-archive.com/data/#2018

Directory structure:

dataset/
├── ISIC2018_Task1-2_Training_Input/
├── ISIC2018_Task1_Training_GroundTruth/
├── ISIC2018_Task1-2_Validation_Input/
├── ISIC2018_Task1_Validation_GroundTruth/
├── ISIC2018_Task1-2_Test_Input/
└── ISIC2018_Task1_Test_GroundTruth/


## Project Structure

.
├── dataset/
├── output/
├── visualizations/
├── train_model.py
├── evaluate_model.py
├── analyze_predictions.py
├── plot_losses.py
├── requirements.txt
├── report.tex
├── report.pdf
├── README.md
└── .gitignore


## Setup and Installation

```bash
git clone https://github.com/yasirjumani/Initial-commit-for-Medical-Image-Segmentation-Project.git
cd Initial-commit-for-Medical-Image-Segmentation-Project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt



## Usage
Train the Model
python train_model.py
Evaluate the Modelpython evaluate_model.py
Generate Uncertainty Maps
python analyze_predictions.py

Results
Quantitative and qualitative results, including Dice and IoU scores and uncertainty visualizations, are detailed in the technical report.

Contributing
Issues and pull requests are welcome.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
GitHub: yasirjumani
