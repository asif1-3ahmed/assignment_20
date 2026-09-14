import pandas as pd
import numpy as np
import streamlit as st

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# TEXT CLEANING
# ==========================================

def clean_text(text):

    text = str(text).lower()

    text = "".join(
        char if char.isalpha() or char.isspace()
        else " "
        for char in text
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in ENGLISH_STOP_WORDS
    ]

    return " ".join(words)


# ==========================================
# LOAD DATA + TF-IDF
# ==========================================

@st.cache_data
def load_data():

    df = pd.read_csv("tmdb_5000_movies.csv")

    # Handle missing overviews
    df["overview"] = df["overview"].fillna("")

    # Clean text
    df["clean_text"] = df["overview"].apply(clean_text)

    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = tfidf.fit_transform(
        df["clean_text"]
    )

    return df, tfidf_matrix


# Load once and cache
df, tfidf_matrix = load_data()


# ==========================================
# RECOMMENDATION FUNCTION
# ==========================================

def recommend(item_name, top_n=5):

    matches = df.index[
        df["title"].str.lower() == item_name.lower()
    ].tolist()

    if not matches:
        return pd.DataFrame(
            columns=["title", "similarity"]
        )

    item_index = matches[0]

    # Calculate similarity ONLY for selected movie
    scores = cosine_similarity(
        tfidf_matrix[item_index],
        tfidf_matrix
    ).flatten()

    # Sort highest similarity first
    similar_indices = np.argsort(scores)[::-1]

    # Remove selected movie
    similar_indices = [
        i for i in similar_indices
        if i != item_index
    ]

    # Get top results
    top_indices = similar_indices[:top_n]

    recommendations = pd.DataFrame({
        "title": df.iloc[top_indices]["title"].values,
        "similarity": np.round(
            scores[top_indices],
            4
        )
    })

    return recommendations


# ==========================================
# STREAMLIT INTERFACE
# ==========================================

st.title("🎬 Movie Recommendation System")

st.write(
    "Find movies similar to your selected movie "
    "using content-based recommendation."
)


# Movie dropdown
movie_list = sorted(
    df["title"].dropna().unique()
)

selected_movie = st.selectbox(
    "Select a movie:",
    movie_list
)


# Number of recommendations
top_n = st.slider(
    "Number of recommendations:",
    min_value=1,
    max_value=10,
    value=5
)


# Recommendation button
if st.button("🎯 Get Recommendations"):

    with st.spinner("Finding similar movies..."):

        recommendations = recommend(
            selected_movie,
            top_n
        )

    st.subheader(
        f"Movies similar to {selected_movie}"
    )

    if recommendations.empty:

        st.error("Movie not found.")

    else:

        for i, row in recommendations.iterrows():

            st.write(
                f"### {i + 1}. {row['title']}"
            )

            st.write(
                f"Similarity Score: "
                f"{row['similarity']}"
            )