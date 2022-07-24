# Movie Recommender

## Introduction
A movie recommender app that provides four different types of recommendation based on user input

## Description
The webapp is built using Streamlit and users can choose the type and number of movie recommendations they would like from the menu on the sidebar.<br>
<a href="https://presthisbutton-movie-recommender-recommender-streamlit-vfjxtn.streamlitapp.com/" target="_blank">Movie Recommender App</a>

The first option is a popularity based recommendation, in which the returned results are based on user ratings and the number of ratings a movie has recieved. 

The second option utilizes item based collaborative filtering. Users can select a movie they like and the recommender will suggest movies that are similar based on the Pearson correlation coefficient (R) between them.

The third option utilizes user based collaborative filtering. The cosine similarity between all users are calculated. Once the user select their user ID, the recommender will suggest movies that other similar users have enjoyed. 

The fourth option suggests a random movie from the database to the user.

## Tools
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
