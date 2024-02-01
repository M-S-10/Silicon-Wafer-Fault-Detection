# Silicon-Wafer-Fault-Detection-Project

## Overview
This project focuses on the detection of silicon wafer quality using sensor data. The classification task involves predicting whether a silicon wafer is "good" or "bad" based on the values from 590 sensors.

## Motivation
The semiconductor industry demands efficient methods for quality control in the production of silicon wafers. Traditional inspection methods are often time-consuming and manual. This project aims to automate the detection process using machine learning techniques, specifically the k-Nearest Neighbors (KNN) algorithm.

## Objective
The primary objective of this project is to develop a robust and accurate model that can classify silicon wafers as "good" or "bad" based on sensor readings. The goal is to streamline the quality control process and reduce the reliance on manual inspection, leading to increased efficiency and productivity in semiconductor manufacturing.

## Methodology
1. **Data Preprocessing:**
   - Load the training dataset containing 590 sensors and labels.
   - Handle missing values using the SimpleImputer.
   - Split the dataset into training and testing sets.

2. **Model Training:**
   - Utilize the k-Nearest Neighbors (KNN) algorithm for classification.
   - Train the model on the training dataset.

3. **Testing and Evaluation:**
   - Test the trained model on a new dataset.
   - Evaluate the performance using metrics such as confusion matrix and classification report.

## Results
The trained KNN model demonstrated promising results in classifying silicon wafers based on sensor data. Detailed evaluation metrics, including precision, recall, and F1-score, are provided in the classification report.

## Project Structure
- **main.py**: Main script for training the KNN classifier and making predictions on a new dataset.
- **your_training_dataset.csv**: CSV file containing the training dataset with 590 sensors and labels.
- **your_testing_dataset.csv**: CSV file for testing the trained model on a new dataset.
- **.venv/**: Virtual environment directory (you can customize this based on your virtual environment setup).
- **README.md**: Project documentation.

## Requirements
- Python 3.x
- Required Python packages: pandas, numpy, scikit-learn
- Raw DataSet -> Wafer-dataset on #kaggle via @KaggleDatasets https://www.kaggle.com/datasets/himanshunayal/waferdataset?utm_medium=social&utm_campaign=kaggle-dataset-share&utm_source=twitter 

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Silicon-Wafer-Detection.git
   cd Silicon-Wafer-Detection
