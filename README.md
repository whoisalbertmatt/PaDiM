# Industrial Visual Anomaly Detection using PaDiM

## Summary 
- Built with PyTorch
- PaDiM-inspired anomaly detection
- Spatial feature modelling
- Mahalanobis distance scoring
- Heatmap visualisation
- 98.8% Accuracy
- 100% Recall

## Introduction
Quality inspection is an essential part of modern manufacturing. Companies often spend significant time and resources identifying defective products before they reach customers.

Traditional rule-based vision systems can struggle with unforeseen defects because they rely on predefined patterns. Anomaly detection takes a different approach: it learns what a normal, non-defective product looks like and flags samples that deviate from that normal distribution.

## Objective
The objective of this project was to build a computer vision anomaly detection pipeline capable of identifying defective product images using only normal images during training.

A reconstruction-based autoencoder was initially explored. While it was able to reconstruct normal images, it struggled to clearly separate defective samples from normal ones. This motivated the transition to a feature-distribution approach inspired by PaDiM.

## Dataset
This project uses the Bottle category from the MVTec AD dataset.

- Training data: normal bottle images only
- Test data: normal and defective bottle images
- Task: detect whether an image is normal or anomalous and generate a heatmap showing suspicious regions

## Method
The system uses a pretrained ResNet18 model to extract spatial feature maps from images. Normal feature distributions are modelled at each spatial location using mean and covariance statistics. During inference, Mahalanobis distance is used to measure how far a test image deviates from the learned normal distribution.

## Pipeline
1. Load MVTec AD bottle images
2. Extract ResNet18 spatial feature maps
3. Select feature channels for stable covariance estimation
4. Compute spatial mean and covariance from normal images
5. Score test images using Mahalanobis distance
6. Classify images using a threshold
7. Generate anomaly heatmaps

## Results
| Metric | Score |
|---|---:|
| Accuracy | 98.8% |
| Precision | 98.4% |
| Recall | 100% |
| F1 Score | 99.2% |
| ROC-AUC | 99.7% |

The model achieved 100% recall on the Bottle test set, meaning no defective samples were missed in this experiment.

## To Run
Train PaDiM statistics:

```bash
python -m src.padim.train_padim
```
## To Evaluate
```bash
python -m src.padim.evaluate_padim
```
## For Heatmap
```bash
python -m src.padim.heatmap
```
## Limitations
- Currently tested only on the Bottle category.
- The dataset and saved model/statistics files are not included in this repository.
- Generalisation to other object categories requires retraining.
- Performance may vary under different lighting, viewpoints, and backgrounds.

## Future Work
- Future Work
- Add FastAPI inference endpoint
- Add Docker support
- Support multiple MVTec object categories
- Add command-line image inference
- Compare with PatchCore or autoencoder-based methods
- Deploy as a simple web demo
