import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity

# a function that returns the top n movies based on ratings
def top_movies(n):
    top_movie = movie_ratings.groupby("title")[["rating", "rating_count"]].mean().sort_values(by = "rating_count",
                ascending = False).head(n) 
    results = top_movie.merge(movie_ratings[["title", "genres", "year"]], left_index = True, 
                              right_on = "title").drop_duplicates(subset = "title")
    results.drop(columns = ["rating", "rating_count"], inplace = True)
    results.reset_index(drop = True, inplace = True)
    return results

# a function that returns the n most similar movies to the selected title
def similar_movies(title, n):
    # get the rating of the input movie
    movie_rating = movie_pivot[title]
    # find the correlation between the input movie and all other movies
    similar_movie = movie_pivot.corrwith(movie_rating)
    
    # create a dataframe with all correlation values and drop all Nans
    movie_corr = pd.DataFrame(similar_movie, columns = ["PearsonR"])
    movie_corr.dropna(inplace = True)
    
    # merge the correlation dataframe with the movrat dataframe which has title and rating count information
    movie_corr_summary = movie_corr.merge(movie_ratings[["title", "rating_count"]], left_index = True, 
                                          right_on = "title")
    
    # drop the input movie itself from the dataframe
    movie_corr_summary.drop(movie_corr_summary[movie_corr_summary["title"] == title].index, 
                            inplace = True)
    
    # filter out movies with less than 10 ratings
    filtered_movie = movie_corr_summary[movie_corr_summary["rating_count"] >= 10]
    
    # group the movies by title and sort them by R score then rating count
    top_similar_movie = filtered_movie.groupby("title")[["PearsonR", "rating_count"]].mean().sort_values(
        by = ["PearsonR", "rating_count"], ascending = [False, False]).head(n)
    
    # merge the dataframe with the movie ratings dataframe to get genres and year of the movies
    results = top_similar_movie.merge(movie_ratings[["title", "genres", "year"]], left_index = True, 
                                      right_on = "title").drop_duplicates(subset = "title")
    results.drop(columns = ["PearsonR", "rating_count"], inplace = True)
    results.reset_index(drop = True, inplace = True)
    
    return results

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
    
    # merge the dataframe with the movie ratings dataframe to get genres and year of the movies
    results = recommendation.merge(movie_ratings[["title", "genres", "year"]], left_index = True, 
                                   right_on = "title").drop_duplicates(subset = "title")
    results.drop(columns = ["predicted_rating"], inplace = True)
    results.reset_index(drop = True, inplace = True)
    
    return results

# a function that returns a random movie
def random_movie():
    n = random.randint(1, len(movies))
    return movies[movies.index == n][["title", "genres", "year"]]


# read csv files and create dataframe
movies = pd.read_csv("Data/movies.csv")
ratings = pd.read_csv("Data/ratings.csv")

# split the year from the title to a different column and clean the columns
movies[["title", "year", "0", "1", "2"]] = movies["title"].str.split("(", expand = True)
movies["title"] = movies["title"].str.rstrip()
movies["year"] = movies["year"].str.strip(")")
movies.drop(columns = ["0", "1", "2"], inplace = True)

movie_ratings = movies.merge(ratings)

# add a column with rating count
movie_ratings["rating_count"] = movie_ratings.groupby("title")["title"].transform("count")

# split the year from the title to a different column and clean the columns
# movie_ratings[["title", "year", "0", "1", "2"]] = movie_ratings["title"].str.split("(", expand = True)
# movie_ratings["title"] = movie_ratings["title"].str.rstrip()
# movie_ratings["year"] = movie_ratings["year"].str.strip(")")
# movie_ratings.drop(columns = ["0", "1", "2"], inplace = True)

# create pivot tables for movie and user based recommendations
movie_pivot = pd.pivot_table(data = movie_ratings, values = "rating", index = "userId", columns = "title")
movie_pivot_filled = movie_pivot.fillna(0)

movie_list = movie_ratings["title"].unique()

# Streamlit app elements
st.title("Movie Recommender")

st.sidebar.markdown("## Choose a recommender")
option = st.sidebar.selectbox("", ["Highly rated", "Movie based", "User based", "Surprise me"])

if option == "Highly rated":
    # Recommendation based on movie ratings
    top_movies_num = st.sidebar.slider("Number of recommendations", min_value = 1, max_value = 100)
    top_movies_recommendation = top_movies(top_movies_num)
    st.write("Here are the top movies based on user ratings")
    st.table(top_movies_recommendation)

elif option == "Movie based":
    # Recommendation based on movies the users like
    movie_name = st.sidebar.selectbox("Choose a movie", movie_list)
    similar_movies_num = st.sidebar.slider("Number of recommendations", min_value = 1, max_value = 100)
    movie_based_recommendation = similar_movies(movie_name, similar_movies_num)
    st.write("Here are the recommended movies that are similar to the one you have chosen")
    st.table(movie_based_recommendation)

elif option == "User based":
    # Recommendation based on what similar users like
    userID = st.sidebar.selectbox("Choose a user ID", range(1, 611))
    recommended_movies_num = st.sidebar.slider("Number of recommendations", min_value = 1, max_value = 100)
    user_based_recommendation = recommended_movies(userID, recommended_movies_num)
    st.write("Here are the recommended movies based on what similar users to the chosen one have enjoyed")
    st.table(user_based_recommendation)
    
elif option == "Surprise me":
    # Recommend a random movie
    clicked = st.sidebar.button("Show me a movie")
    if clicked:
        st.table(random_movie())
    














