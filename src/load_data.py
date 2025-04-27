import pandas as pd

def load_movie_data(moviecsv, creditcsv):
    movies = pd.read_csv(moviecsv)
    credits = pd.read_csv(creditcsv)

    credits = credits.rename(columns={"movie_id": "id"})
    df = movies.merge(credits, on='id')

    return df