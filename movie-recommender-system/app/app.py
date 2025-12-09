import streamlit as st
import pickle
import pandas as pd
import requests
import os
from dotenv import load_dotenv
import gdown

# IMPORTANT: You must provide your own TMDB API key in the .env file as API_KEY="your_api_key".
# Without a valid API key, movie posters will not be displayed.

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")

def fetch_poster(movie_id):
    response = requests.get(
        f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}'
    )
    data = response.json()
    poster_path = data.get('poster_path')
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    else:
        return ""

def recommend(movie_name):
    index = movies[movies['title'] == movie_name].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movies_posters = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommended_movies_posters

movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

# Google Drive file ID for similarity.pkl
SIMILARITY_FILE_ID = "1tE3A7vlhFsSnciiptK6DoiSBAtNwXZIT"
SIMILARITY_FILE_PATH = "similarity.pkl"

# Check if similarity.pkl exists locally, else download from Google Drive
if not os.path.exists(SIMILARITY_FILE_PATH):
    gdown.download(
        f"https://drive.google.com/uc?id={SIMILARITY_FILE_ID}",
        SIMILARITY_FILE_PATH,
        quiet=False
    )

similarity = pickle.load(open(SIMILARITY_FILE_PATH, 'rb'))

# Sidebar for instructions and credits
st.sidebar.title("🎬 Movie Recommender")
st.sidebar.markdown(
    """
    **How to use:**  
    1. Select a movie from the dropdown.  
    2. Click 'Recommend' to view similar movies.  
    3. Make sure you have set your TMDB API key in `.env` file.

    ---
    **Credits:**  
    Made with ❤️ using Streamlit and TMDB API.
    """
)

# Main page header and description
st.markdown(
    """
    <h1 style='text-align: center; color: #FFF8E7;'>Movie Recommender System 🍿</h1>
    <p style='text-align: center; color: #333333; font-size:18px;'>
        Discover movies similar to your favorites!  
        Powered by machine learning and TMDB.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

selected_movie_name = st.selectbox(
    '🎥 **Select a movie to get recommendations:**',
    movies['title'].values
)

if st.button('🚀 Recommend'):
    with st.spinner('Finding your next favorite movies...'):
        names, posters = recommend(selected_movie_name)
    st.markdown("#### Recommended Movies")
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            st.image(
                posters[idx],
                caption=names[idx],
                use_container_width=True,
                output_format="JPEG"
            )
            st.markdown(
                f"<div style='text-align: center; color: #FFF8E7; font-weight: bold;'>{names[idx]}</div>",
                unsafe_allow_html=True
            )