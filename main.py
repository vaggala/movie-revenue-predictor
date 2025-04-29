from src.load_data import load_movie_data
from src.preprocessing import preprocess_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model



def main():
    movies_df = load_movie_data('data/tmdb_5000_movies.csv', 'data/tmdb_5000_credits.csv')
    
    processed_df = preprocess_data(movies_df)
    
    model, X_test, y_test, y_pred = train_model(processed_df)
    
    metrics = evaluate_model(y_test, y_pred, undo_log=True)
    
if __name__ == "__main__":
    main()