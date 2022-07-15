import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("Data/movies.csv")
ratings = pd.read_csv("Data/ratings.csv")
movie_ratings = movies.merge(ratings)
movie_ratings["rating_count"] = movie_ratings.groupby("title")["title"].transform("count")
movie_ratings[["title", "year", "0", "1", "2"]] = movie_ratings["title"].str.split("(", expand = True)
movie_ratings["title"] = movie_ratings["title"].str.rstrip()
movie_ratings["title"] = movie_ratings["title"].str.lower()
movie_ratings["year"] = movie_ratings["year"].str.strip(")")
movie_ratings.drop(columns = ["0", "1", "2"], inplace = True)

movie_pivot = pd.pivot_table(data = movie_ratings, values = "rating", index = "userId", columns = "title")

movie_pivot_filled = movie_pivot.fillna(0)

def top_movies(n):
    top_movie = movie_ratings.groupby("title")[["rating", "rating_count"]].mean().sort_values(
        by = "rating_count", ascending = False).head(n)   
    return top_movie.index.str.title()

# a function that returns the n most similar movies to the selected title
def similar_movies(title, n):
    # get the rating of the input movie
    try:
        movie_rating = movie_pivot[title.lower()]
        # find the correlation between the input movie and all other movies
        similar_movie = movie_pivot.corrwith(movie_rating)
    
        # create a dataframe with all correlation values and drop all Nans
        movie_corr = pd.DataFrame(similar_movie, columns = ["PearsonR"])
        movie_corr.dropna(inplace = True)
    
        # merge the correlation dataframe with the movrat dataframe which has title and rating count information
        movie_corr_summary = movie_corr.merge(movie_ratings[["title", "rating_count"]], left_index = True, 
                                              right_on = "title")
    
        # drop the input movie itself from the dataframe
        movie_corr_summary.drop(movie_corr_summary[movie_corr_summary["title"] == title.lower()].index, 
                                inplace = True)
    
        # filter out movies with less than 10 ratings
        filtered_movie = movie_corr_summary[movie_corr_summary["rating_count"] >= 10]
    
        # group the movies by title and sort them by R score then rating count
        top_similar_movie = filtered_movie.groupby("title")[["PearsonR", "rating_count"]].mean().sort_values(
            by = ["PearsonR", "rating_count"], ascending = [False, False]).head(n)
    
        return top_similar_movie.index.str.title()
    
    except:
        error_msg = "No movies found"
        return error_msg

# a function that returns the top n movies to the user based on what other similar users like
def recommended_movies(userID, n):
    # create a dataframe of cosine similarities between all users
    user_similarities = pd.DataFrame(cosine_similarity(movie_pivot_filled), columns = movie_pivot_filled.index, 
                                     index = movie_pivot_filled.index)
    
    # calculate the weight of all other users based on the similarity between them and the input user
    weights = (user_similarities.query("userId!=@userID")[userID] / sum(user_similarities.query(
        "userId!=@userID")[userID]))
    
    # select the movies that the input user has not watched
    unrated_movies = movie_pivot_filled.loc[movie_pivot_filled.index != userID, 
                                            movie_pivot_filled.loc[userID,:] == 0]
    
    # calculate the dot product between the unrated movies and the weights
    rating_estimates = pd.DataFrame(unrated_movies.T.dot(weights), columns = ["predicted_rating"])
    
    recommendation = rating_estimates.sort_values("predicted_rating", ascending = False).head(n)
    
    return recommendation.index.str.title()

# Streamlit app elements
st.title("Movie Recommender")

# Recommendation based on movie ratings
top_movies_num = st.slider("How many top movie recommendations would you like?", min_value = 1, max_value = 100)
top_movies_recommendation = top_movies(top_movies_num)
st.write(top_movies_recommendation)

# Recommendation based on movies the users like
movie_name = st.text_input("Please enter a movie you like")
similar_movies_num = st.slider("How many similar movie recommendations would you like?", min_value = 1, max_value = 100)
movie_based_recommendation = similar_movies(movie_name, similar_movies_num)
st.write(movie_based_recommendation)

# Recommendation based on what similar users like
userID = st.number_input("Please enter your user ID", min_value = 1)
recommended_movies_num = st.slider("How many movie recommendations would you like based on what other users like you have enjoyed", min_value = 1, max_value = 100)
user_based_recommendation = recommended_movies(userID, recommended_movies_num)
st.write(user_based_recommendation)










