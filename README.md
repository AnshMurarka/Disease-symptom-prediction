# Disease-symptom-prediction

This repository contains a Python-based implementation of a disease prediction system. 
It leverages machine learning with a Random Forest Classifier to predict diseases based on user-provided symptoms. 
The system uses symptom severity weights and provides actionable precautions for the identified diseases.

# Dataset source: 
The data is taken from kaggle website "Disease Symptom Prediction"(https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)

# Features
->*Custom Predictions*: Accepts symptoms as input and predicts the disease along with associated confidence levels and precautionary measures.
->*Data Visualization*: Visualizes missing values in the dataset before cleaning.
->*Symptom Preprocessing*: Maps symptoms to predefined severity weights.
->*Machine Learning Model*: Trains a Random Forest Classifier to predict diseases.
->*Evaluation Metrics*: Displays model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

# Datasets
The project uses the following datasets:

=>Symptom-severity.csv - Contains symptoms and their severity weights.
->dataset.csv - Lists diseases and associated symptoms.
->symptom_Description.csv - Descriptions of diseases.
->symptom_precaution.csv - Precautionary measures for diseases.

# Usage
Clone the repository and ensure all datasets are in the specified directory.Run the Python script to:
->Train the model.
->Evaluate the model's performance.
->Input symptoms manually using the predd function for disease predictions and precautions.
