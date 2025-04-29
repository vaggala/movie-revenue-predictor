# Movie Revenue Prediction

This project develops a machine learning model to predict box office revenue based on movie metadata.  
The model uses Random Forest Regression combined with structured feature engineering and Natural Language Processing (NLP) for text data.

## Project Overview

- Model: Random Forest Regressor (scikit-learn)
- Structured Features: Budget, cast size, runtime, genres, popularity, vote average, vote count
- Unstructured Feature (NLP): Overview sentiment analysis using Hugging Face Transformers
- Performance:  
  - R² Score: 0.678 (68% of revenue variability explained)
  - Metrics: MAE, RMSE
- Visualizations: 
  - Actual vs Predicted Revenue (normal scale)

## Tech Stack

- Python
- scikit-learn
- pandas, numpy
- matplotlib
- Hugging Face Transformers

## Future Improvements

- Model saving with `joblib` for faster reruns
- Add log-scaled revenue visualization
- Experiment with other regression models (e.g., Gradient Boosting, XGBoost)
- Expand feature engineering (e.g., director popularity, production company features)
- Tune hyperparameters for better predictive performance


