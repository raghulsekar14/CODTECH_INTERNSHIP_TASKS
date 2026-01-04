import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# -------------------------------
# App Config
# -------------------------------
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommendation System")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    ratings_path = os.path.join(BASE_DIR, "ratings.csv")
    movies_path = os.path.join(BASE_DIR, "movies.csv")
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    ratings = ratings[['userId', 'movieId', 'rating']]
    return ratings, movies

ratings, movies = load_data()

# -------------------------------
# Create User–Item Matrix
# -------------------------------
user_item_matrix = ratings.pivot(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0).astype(float)

# -------------------------------
# Train SVD Model
# -------------------------------
@st.cache_data
def train_svd(matrix, n_components=50):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    latent_matrix = svd.fit_transform(matrix)
    predicted_ratings = np.dot(latent_matrix, svd.components_)
    return predicted_ratings

predicted_ratings = train_svd(user_item_matrix)

# -------------------------------
# User Selection
# -------------------------------
user_ids = user_item_matrix.index.tolist()
movie_ids = user_item_matrix.columns.tolist()

user_id = st.selectbox("Select User ID", user_ids)

# -------------------------------
# Generate Recommendations
# -------------------------------
if st.button("Recommend Movies"):
    user_index = user_ids.index(user_id)
    user_predictions = predicted_ratings[user_index]

    watched_movies = ratings[ratings['userId'] == user_id]['movieId'].values

    recommendations = [
        (movie_id, user_predictions[idx])
        for idx, movie_id in enumerate(movie_ids)
        if movie_id not in watched_movies
    ]

    top_movies = sorted(recommendations, key=lambda x: x[1], reverse=True)[:5]

    st.subheader("⭐ Top 5 Recommended Movies")
    for movie_id, score in top_movies:
        title = movies[movies['movieId'] == movie_id]['title'].values[0]
        st.write(f"**{title}** — Predicted Rating: {round(score, 2)}")
