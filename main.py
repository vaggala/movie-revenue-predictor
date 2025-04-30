import os
import joblib
from src.load_data import load_movie_data
from src.preprocessing import preprocess_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.visualize_model import plot_actual_vs_predicted, plot_feature_importance

def main():
    model_path = "models/movie_revenue_model.joblib"
    
    if os.path.exists(model_path):
        print("Model already exists. Loading...")
        model = joblib.load(model_path)
    else:
        print("Model not found. Training...")
        movies_df = load_movie_data('data/tmdb_5000_movies.csv', 'data/tmdb_5000_credits.csv')
        processed_df = preprocess_data(movies_df)
        model, X_test, y_test, y_pred = train_model(processed_df)
        evaluate_model(y_test, y_pred, undo_log = True)
        
        plot_actual_vs_predicted(y_test, y_pred, save_path = "visualizations/normal_scale_plot.png")
        plot_feature_importance(model, X_test.columns, save_path = "visualizations/feature_importance.png")

if __name__ == "__main__":
    main()
