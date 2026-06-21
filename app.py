import pickle
import streamlit as st
import requests


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)


# ── Load data once ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    with open('model/movie_list.pkl', 'rb') as f:
        movies = pickle.load(f)
    with open('model/similarity.pkl', 'rb') as f:
        similarity = pickle.load(f)
    movies = movies.reset_index(drop=True)   # align label index == positional index
    return movies, similarity


# ── Poster fetcher ────────────────────────────────────────────────────────────
FALLBACK = "https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg"
API_KEY  = "02c30b6da53a300ba99914e5b6b8a5a2"   # replace with your TMDB v3 key

def fetch_poster(movie_id: int) -> str:
    try:
        url  = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
        data = requests.get(url, timeout=5).json()
        path = data.get("poster_path")
        if path:
            return f"https://image.tmdb.org/t/p/w500{path}"
    except Exception:
        pass
    return FALLBACK


# ── Recommend function ────────────────────────────────────────────────────────
def recommend(movie: str, movies, similarity) -> tuple[list, list]:
    """Return (names, poster_urls) for top-5 similar movies."""
    names, posters = [], []

    matches = movies[movies['title'] == movie]
    if matches.empty:
        st.error(f'Movie "{movie}" not found.')
        return names, posters

    idx       = matches.index[0]                                     # positional after reset
    distances = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)

    for i, _ in distances[1:6]:
        names.append(movies.iloc[i]['title'])
        posters.append(fetch_poster(movies.iloc[i]['movie_id']))

    return names, posters


# ── UI ────────────────────────────────────────────────────────────────────────
st.title('🎬 Movie Recommender System')
st.caption('Content-Based Filtering · TMDB 5000 Dataset · TF-IDF + Cosine Similarity')

movies, similarity = load_data()

selected_movie = st.selectbox(
    "Type or select a movie",
    movies['title'].values
)

if st.button('Show Recommendations', type='primary'):
    with st.spinner('Finding similar movies...'):
        recommended_movie_names, recommended_movie_posters = recommend(
            selected_movie, movies, similarity
        )

    if recommended_movie_names:
        cols = st.columns(5)
        for col, name, poster in zip(cols, recommended_movie_names, recommended_movie_posters):
            with col:
                st.image(poster)
                st.caption(name)
