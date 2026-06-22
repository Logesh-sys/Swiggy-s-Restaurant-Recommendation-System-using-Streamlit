import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Restaurant Recommender", layout="wide")

# =========================
# CLEAN UI
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
}
section[data-testid="stSidebar"] {
    background: #f0f2f6 !important;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 8px;
    padding: 10px;
    width: 100%;
}
.stButton>button:hover {
    background-color: #e60000;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("final_data.csv")

encoder = pickle.load(open("encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# =========================
# RECREATE FINAL MATRIX
# =========================
model_df = df.drop(columns=['name'])

encoded_cat = encoder.transform(model_df[['city', 'cuisine']])

numerical_cols = ['effective_rating', 'rating_count', 'cost', 'no_rating']
scaled_num = scaler.transform(model_df[numerical_cols])

final_df = hstack([encoded_cat, csr_matrix(scaled_num)]).tocsr()

# =========================
# TITLE
# =========================
st.title("🍽️ Restaurant Recommendation System")
st.write("Find restaurants based on your preferences")

# =========================
# INPUTS
# =========================
col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("Select City", sorted(df['city'].unique()))
    cuisine = st.selectbox("Select Cuisine", sorted(df['cuisine'].unique()))

with col2:
    cost = st.slider("Cost", 100, 450, 300, key="cost_slider")
    rating = st.slider("Rating", 1.0, 5.0, 4.0, key="rating_slider")

# =========================
# RECOMMEND FUNCTION
# =========================
def recommend_restaurants(city, cuisine, cost, rating, top_n=10):

    # STRICT FILTER (used for alert check)
    strict_df = df[
        (df['city'] == city) &
        (df['rating'] >= rating) &
        (df['cost'] == cost)
    ]

    # Step 1: Strict filter
    filtered_df = df[
        (df['city'] == city) &
        (df['rating'] >= rating - 0.2) &
        (df['cost'] >= cost - 50) &
        (df['cost'] <= cost + 50)
    ]

    # Step 2: Relax
    if len(filtered_df) < top_n:
        filtered_df = df[
            (df['city'] == city) &
            (df['rating'] >= rating - 0.5) &
            (df['cost'] >= cost - 150) &
            (df['cost'] <= cost + 150)
        ]

    # Step 3: Relax more
    if len(filtered_df) < top_n:
        filtered_df = df[
            (df['city'] == city) &
            (df['rating'] >= rating - 1.0)
        ]

    # Step 4: Final fallback
    if len(filtered_df) < top_n:
        filtered_df = df[df['city'] == city]

    filtered_indices = filtered_df.index

    # Input vector
    input_df = pd.DataFrame({
        'city': [city],
        'cuisine': [cuisine],
        'rating': [rating],
        'rating_count': [50],
        'cost': [cost],
        'no_rating': [0]
    })

    input_df['effective_rating'] = input_df['rating'] * (
        input_df['rating_count'] / (input_df['rating_count'] + 50)
    )

    encoded_cat = encoder.transform(input_df[['city', 'cuisine']])

    scaled_num = scaler.transform(
        input_df[['effective_rating','rating_count','cost','no_rating']]
    )

    input_vector = hstack([encoded_cat, csr_matrix(scaled_num)]).tocsr()

    # Similarity
    filtered_matrix = final_df[filtered_indices]
    scores = cosine_similarity(input_vector, filtered_matrix).flatten()

    # Rank
    top_indices = filtered_indices[scores.argsort()[::-1][:top_n]]

    return df.loc[top_indices][['name','city','cuisine','rating','cost']], strict_df

# =========================
# BUTTON ACTION
# =========================
if st.button("🔍 Recommend Restaurants"):

    results, strict_df = recommend_restaurants(city, cuisine, cost, rating)

    # ✅ ALERT FIX
    if strict_df.empty:
        st.warning(
            "⚠️ No exact match found.\n\n"
            "👉 Showing closest available restaurants."
        )

    st.subheader("Top Recommendations")

    results = results.sort_values(by='rating', ascending=False)

    st.dataframe(results, use_container_width=True)