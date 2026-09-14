#PART 4 — Simple App Interface
#Task 6: Streamlit App
import pandas as pd
import numpy as np

import streamlit as st

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)

from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------
# Load Dataset
# -----------------------------------

df = pd.read_csv("tmdb_5000_movies.csv")


# -----------------------------------
# Clean Text
# -----------------------------------

df["overview"] = df["overview"].fillna("")


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


df["clean_text"] = df["overview"].apply(
    clean_text
)


# -----------------------------------
# TF-IDF
# -----------------------------------

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf.fit_transform(
    df["clean_text"]
)


# -----------------------------------
# Similarity
# -----------------------------------

similarity_matrix = cosine_similarity(
    tfidf_matrix
)


# -----------------------------------
# Recommendation Function
# -----------------------------------

def recommend(item_name, top_n=5):

    matches = df.index[
        df["title"].str.lower() == item_name.lower()
    ].tolist()

    if not matches:
        return pd.DataFrame(
            columns=["title", "similarity"]
        )

    item_index = matches[0]

    scores = similarity_matrix[item_index]

    similar_indices = np.argsort(
        scores
    )[::-1]

    similar_indices = [
        i
        for i in similar_indices
        if i != item_index
    ]

    top_indices = similar_indices[:top_n]

    recommendations = pd.DataFrame({
        "title": df.iloc[top_indices]["title"].values,
        "similarity": np.round(
            scores[top_indices],
            4
        )
    })

    return recommendations


# -----------------------------------
# Streamlit UI
# -----------------------------------

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


# Recommendation button
if st.button("Get Recommendations"):

    recommendations = recommend(
        selected_movie,
        top_n=5
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