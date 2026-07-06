# Industrial Visual Anamoly Detection using PaDiM

## Intro
Quality check is an essential and inevitable part of every production. Hundreds of people are involved in production to identify and rectify defects.
Companies also spend a fortune to run and maintain the quality control. Traditional vision based techniques struggle with unforseen defects. Whereas, anomaly
detection learns the good, non-defective product and flags when there appears a defective one. 

## Objectives
Primary objective was to identify and flag defects. Reconstruction-based Autoencoder was intially used to resolve the probelm. It was successful in reconstructing 
the normal images, however this struggled to distinguish defective samples. This motivated the transition to a feature-distribution approach inspired by PaDiM.

## Results 
- Accuracy: 98.8%
- Precision: 98.4%
- Recall: 100%
- F1 Score: 99.2%
- ROC-AUC: 99.7%

## Method
The system uses a pretrained ResNet18 model to extract spatial feature maps from normal images. It models the normal feature distribution at each spatial location using mean and covariance statistics, then computes Mahalanobis distance to identify anomalous regions.

## Pipeline

1. Load MVTec AD images
2. Extract ResNet18 feature maps
3. Select feature channels
4. Compute spatial mean and covariance
5. Score test images using Mahalanobis distance
6. Generate anomaly heatmaps

## Run

```bash
python -m src.padim.train_padim
python -m src.padim.evaluate_padim
python -m src.padim.heatmap
