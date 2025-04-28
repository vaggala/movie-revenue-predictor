import pandas as pd
import numpy as np
import ast

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# loading hugging face model and tokenizer for use. 
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

def get_sentiment(text):
    try:
        if not isinstance(text, str) or len(text.strip()) == 0: # just checks if the text is a non-valid empty string
            return 0
        inputs = tokenizer(text, return_tensors = "pt", truncation = True) # retruns text as pytorch tensors so model can understand text, also cuts down text if too long so model doesn't crash
        outputs = model(**inputs) ## '**' just take each key value from the dictionary (inputs) as a separate named input for the model
        probs = torch.nn.functional.softmax(outputs.logits, dim=1) # logits are raw scores that will get turned into probabilities. logits are random big numbers and softmaxxing turns them into probabilities that add up to 1
        # probs: [negative, neutral, positive]
        score = (-1 * probs[0][0].item()) + (0 * probs[0][1].item()) + (1 * probs[0][2].item()) # score = How likely is "Negative" (probs[0][0]) + How likely is "Neutral" (probs[0][1]) + How likely is "Positive" (probs[0][2])
        return score
    except:
        return 0

def preprocess_data(df):
    df['genres'] = df['genres'].apply(ast.literal_eval) # converts from string to list of dictionaries for easy parsing
    df['cast'] = df['cast'].apply(ast.literal_eval) # converts from string to list of dictionaries for easy parsing
    
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
    
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df = df.dropna(subset=['release_date'])
    runtime_median = df['runtime'].median()
    df['runtime'] = df['runtime'].fillna(runtime_median)
    df['runtime'] = df['runtime'].clip(lower=30, upper=240)  # very minimal % of movies are > 4 hours long, so its best to clip to make sure the model doesnt get too hung up on these outliers.  
    
    ## log-transform the budget and revenue to minimize the effect of outliers when training
    df['log_budget'] = np.log1p(df['budget'])
    df['log_revenue'] = np.log1p(df['revenue'])
    
    # same for outliers in vote count
    df['log_vote_count'] = np.log1p(df['vote_count'])
    
    df['cast_size'] = df['cast'].apply(lambda x: len(x))
    df['num_genres'] = df['genres'].apply(lambda x: len(x))
    df['release_month'] = df['release_date'].dt.month
    
    df['overview_sentiment'] = df['overview'].apply(get_sentiment)
    df['overview_wordcount'] = df['overview'].apply(lambda x: len(str(x).split())) # honestly curious to see if the length of a movie overview has any correlation with revenue. longer descriptions may be more intriguing movies for viewers.

    selected_cols = [
        'log_budget', 
        'cast_size', 
        'release_month', 
        'num_genres', 
        'popularity',
        'runtime',        
        'vote_average',
        'log_vote_count',
        'overview_sentiment',
        'overview_wordcount',  
        'log_revenue'  
    ]
    df = df[selected_cols]
    
    return df