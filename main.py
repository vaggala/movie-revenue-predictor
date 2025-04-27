from src.load_data import load_movie_data

df = load_movie_data('/Users/vasanthaggala/Documents/GitHub/movie-revenue-predictor/data/tmdb_5000_movies.csv', '/Users/vasanthaggala/Documents/GitHub/movie-revenue-predictor/data/tmdb_5000_credits.csv')

print(df.head())