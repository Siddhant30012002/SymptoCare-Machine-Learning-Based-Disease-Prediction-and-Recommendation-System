# SymptoCare-Machine-Learning-Based-Disease-Prediction-and-Recommendation-System
SymptoCare is a machine learning-based system that predicts diseases from symptoms and provides personalized recommendations, including medications, diet plans, precautions, and workouts. It’s designed for emergency situations where quick, accurate information is crucial for effective decision-making and health management.

## Overview
The **Medicine Recommendation System** is designed to provide personalized recommendations for medicines, diets, precautions, and workouts based on the user's symptoms. This project aims to assist users in managing their health by leveraging machine learning techniques to predict suitable remedies and lifestyle adjustments.

## Project Features:
- **Symptoms Input**: Users input their symptoms, and the system predicts the related disease and provides corresponding medication, diet, precautions, and workout suggestions.
- **Personalized Recommendations**: The system offers tailored suggestions based on symptoms to support health management.
- **Easy-to-Use**: Designed with user-friendly input and output processes to make health management more accessible.

## How It Works:
1. **Data**: The system uses multiple CSV files containing data on diet, precautions, workout, and medication for various diseases.
2. **Model**: A machine learning model has been trained using the data to predict the most relevant suggestions for a given set of symptoms.
3. **Model Training**: The model was trained by me, and hyperparameters were fine-tuned to achieve optimal performance. 
4. **Pickle File**: The final trained model has been saved as a pickle file for easy deployment and use in future applications.

## Technologies Used:
- **Python** for data preprocessing and model building
- **Pandas** for data manipulation
- **Scikit-learn** for model training and evaluation
- **Pickle** for saving the trained model
- **streamlit** for the user friendly frontend

## Installation:
1. Clone the repository:
   ```bash
   git clone <repository_url>

