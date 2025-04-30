import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# load the trained model
model = joblib.load("models/movie_revenue_model.joblib")

# load sentiment model + tokenizer
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# function to calculate sentiment score
def compute_sentiment(text):
    if not text.strip():
        return 0.0
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    # -1 × negative, 0 × neutral, 1 × positive
    score = -probs[0].item() + probs[2].item()
    return score

# month mapping for dropdown
months = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

# full list of genres from dataset
all_genres = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign',
    'History', 'Horror', 'Music', 'Mystery', 'Romance',
    'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'
]

# streamlit ui starts here
st.title("🎬 Movie Revenue Predictor")
st.write("enter details about your movie below to predict box office revenue.")

# form input section
with st.form("prediction_form"):
    budget = st.number_input(
        "Budget ($)",
        min_value=1_000_000,
        value=10_000_000,
        step=500_000,
        help="estimated movie production cost in usd."
    )
    cast_size = st.slider(
        "Cast Size", 1, 50, 10,
        help="number of top-billed cast members."
    )
    release_month = st.selectbox(
        "Release Month", list(months.keys()),
        format_func=lambda x: months[x],
        help="the month the movie was/will be released."
    )
    selected_genres = st.multiselect(
        "Genres",
        options=all_genres,
        help="select all genres that apply to the movie."
    )
    num_genres = len(selected_genres)
    popularity = st.number_input(
        "Popularity Score", min_value=0.0, value=20.0,
        help="tmdb score based on social/search interest. usually between 0 and 100."
    )
    runtime = st.slider(
        "Runtime (minutes)", 30, 240, 120,
        help="duration of the movie in minutes."
    )
    vote_average = st.slider(
        "Vote Average (1–10)", 1.0, 10.0, 7.0,
        help="average tmdb user rating for the movie."
    )
    vote_count = st.number_input(
        "Vote Count", min_value=0, value=500,
        help="total number of user ratings on tmdb."
    )
    overview = st.text_area(
        "Overview (Optional)", "",
        help="short plot description of the movie. used to calculate sentiment score."
    )

    submitted = st.form_submit_button("Predict Revenue")

# prediction section
if submitted:
    log_budget = np.log1p(budget)
    log_vote_count = np.log1p(vote_count)
    overview_wordcount = len(overview.split())
    overview_sentiment = compute_sentiment(overview)

    # build a dataframe instead of numpy array to match training input format
    input_dict = {
        'log_budget': [log_budget],
        'cast_size': [cast_size],
        'release_month': [release_month],
        'num_genres': [num_genres],
        'popularity': [popularity],
        'runtime': [runtime],
        'vote_average': [vote_average],
        'log_vote_count': [log_vote_count],
        'overview_sentiment': [overview_sentiment],
        'overview_wordcount': [overview_wordcount]
    }

    input_df = pd.DataFrame(input_dict)

    # make prediction
    log_pred = model.predict(input_df)[0]
    predicted_revenue = np.expm1(log_pred)

    st.success(f"💰 predicted box office revenue: **${predicted_revenue:,.2f}**")

    with st.expander("🔍 model input breakdown"):
        st.write(f"**selected genres:** {', '.join(selected_genres) if selected_genres else 'None'}")
        st.write(f"**sentiment score:** {overview_sentiment:.3f}")
        st.write(f"**overview word count:** {overview_wordcount}")
        st.write(f"**log(budget):** {log_budget:.2f}")
        st.write(f"**log(vote count):** {log_vote_count:.2f}")
        st.write(f"**release month:** {months[release_month]}")
        st.write(f"**runtime:** {runtime} minutes")
        st.write(f"**vote average:** {vote_average}")

# explainer section
with st.expander("what do the inputs mean?"):
    st.markdown("""
    - **budget**: estimated cost to produce the movie (usd).
    - **cast size**: number of prominent cast members.
    - **release month**: month the movie is/will be released.
    - **genres**: categories the movie fits into (you select them).
    - **popularity score**: tmdb score based on buzz, searches, and social media.
    - **vote average**: average user rating from 1 to 10 on tmdb.
    - **vote count**: number of ratings the movie has on tmdb.
    - **overview**: short movie summary used to calculate sentiment.
    """)
