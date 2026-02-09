# Loan Default Prediction

## Project Overview
This project aims to predict loan defaults using an anonymized dataset. 
The goal is to build a professional ML pipeline following best practices.

## Project Structure

loan-default-ml/
├── app/                  # Scripts or API (future)
├── data/                 # Raw and processed datasets (ignored by Git)
├── models/               # Trained models (future)
├── notebooks/            # Notebooks for analysis and experiments
├── requirements.txt      # Python dependencies
├── README.md             # Project description
├── .gitignore            # Files/folders ignored by Git

## Dataset
The dataset used in this project comes from kaggle and is not included in this repository.

To reproduce the projects:
1. Download the dataset from kaggle
2. Place the CSV file in 'data/raw'

## Config
- src/config.py is used to store constants for the project, such as:
  - TARGET variable (`Default`)
  - Numerical features list
  - Categorical features list