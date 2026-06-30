# Titanic Survival Predictor

## What this project does
A machine learning project that predicts Titanic passenger survival
using three different ML models and compares their performance.

## Models used
- Logistic Regression — 80% accuracy (best)
- KNN — 71% accuracy
- SVM — 65% accuracy

## What I found
- Logistic Regression performed best on this dataset
- SVM underperformed due to unscaled features
- Gender and passenger class were the strongest survival predictors

## Libraries used
- pandas
- scikit-learn
- matplotlib

## How to run
1. Clone the repo
2. Install libraries: `pip install pandas scikit-learn matplotlib`
3. Add train.csv from Kaggle Titanic dataset
4. Run `titanic_ml.py`
