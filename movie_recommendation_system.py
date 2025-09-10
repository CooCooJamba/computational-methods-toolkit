"""
Movie Recommendation System
This script implements a hybrid movie recommendation system using collaborative filtering
and content-based filtering. It uses the Surprise library for SVD-based collaborative
filtering and scikit-learn for content-based recommendations based on movie genres.
"""

import pandas as pd
import os
from surprise import Dataset, Reader, SVD, accuracy, Prediction
from surprise.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    """
    Load movie ratings and movie metadata from CSV files.
    
    Returns:
        tuple: A tuple containing ratings DataFrame and movies DataFrame
    """
    # Construct file paths
    ratings_path = os.path.join('ml-latest-small', 'ratings.csv')
    movies_path = os.path.join('ml-latest-small', 'movies.csv')
    
    # Load data
    ratings_df = pd.read_csv(ratings_path, usecols=['userId', 'movieId', 'rating'])
    movies_df = pd.read_csv(movies_path)
    
    return ratings_df, movies_df


def train_collaborative_filtering(ratings_df):
    """
    Train a collaborative filtering model using SVD algorithm.
    
    Args:
        ratings_df (DataFrame): DataFrame containing user ratings
        
    Returns:
        tuple: A tuple containing trained model and testset for evaluation
    """
    # Load data into Surprise format
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df, reader)
    
    # Split data into training and test sets
    trainset, testset = train_test_split(data, test_size=0.2)
    
    # Create and train SVD model
    model = SVD()
    model.fit(trainset)
    
    return model, testset


def evaluate_collaborative_filtering(model, testset):
    """
    Evaluate the collaborative filtering model using RMSE.
    
    Args:
        model: Trained SVD model
        testset: Test data for evaluation
        
    Returns:
        float: RMSE value
    """
    # Generate predictions
    predictions = model.test(testset)
    
    # Calculate RMSE
    rmse = accuracy.rmse(predictions)
    print(f'RMSE: {rmse}')
    
    return rmse


def create_content_based_recommender(movies_df):
    """
    Create a content-based recommendation system based on movie genres.
    
    Args:
        movies_df (DataFrame): DataFrame containing movie information
        
    Returns:
        tuple: A tuple containing cosine similarity matrix and processed movies DataFrame
    """
    # Process genres
    movies_df = movies_df.copy()
    movies_df['genres'] = movies_df['genres'].str.split('|')
    
    # Create binary genre matrix using OneHotEncoder
    ohe = OneHotEncoder(sparse=False)
    genres_matrix = ohe.fit_transform(
        movies_df['genres'].explode().reset_index(drop=True).str.get_dummies(sep='|')
    )
    
    # Create DataFrame with genre matrix
    genres_matrix = pd.DataFrame(
        genres_matrix,
        columns=ohe.get_feature_names_out(),
        index=movies_df['movieId'].repeat(movies_df['genres'].str.len()).reset_index(drop=True)
    )
    
    # Group by movieId to get final matrix
    genres_matrix = genres_matrix.groupby(genres_matrix.index).sum()
    
    # Apply SVD for dimensionality reduction
    svd = TruncatedSVD(n_components=10)
    svd_matrix = svd.fit_transform(genres_matrix)
    
    # Calculate cosine similarity between movies
    cosine_sim = cosine_similarity(svd_matrix)
    
    return cosine_sim, movies_df


def get_recommendations(movie_id, cosine_sim, movies_df, top_n=5):
    """
    Get content-based movie recommendations for a given movie.
    
    Args:
        movie_id (int): ID of the movie to get recommendations for
        cosine_sim (ndarray): Cosine similarity matrix
        movies_df (DataFrame): DataFrame containing movie information
        top_n (int): Number of recommendations to return
        
    Returns:
        Series: Titles of recommended movies
    """
    # Find index of the movie in the DataFrame
    idx = movies_df[movies_df['movieId'] == movie_id].index[0]
    
    # Get similarity scores for all movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort movies by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get indices of top_n most similar movies (excluding the movie itself)
    sim_indices = [i[0] for i in sim_scores[1:top_n + 1]]
    
    # Return titles of recommended movies
    return movies_df['title'].iloc[sim_indices]


def evaluate_hybrid_system(model, testset, cosine_sim, movies_df):
    """
    Evaluate the hybrid recommendation system using precision and recall.
    
    Args:
        model: Trained SVD model
        testset: Test data for evaluation
        cosine_sim (ndarray): Cosine similarity matrix
        movies_df (DataFrame): DataFrame containing movie information
        
    Returns:
        tuple: A tuple containing precision and recall values
    """
    # Create dictionary for movie ID to title mapping
    movies_df_indexed = movies_df.set_index("movieId")
    movies_dict = movies_df_indexed["title"].to_dict()
    
    # Generate recommendations and predictions
    predictions = []
    
    # Generate recommendations for each user in the test set
    for uid, mid, true_r in testset:
        recommended_movies = get_recommendations(
            movie_id=mid, 
            cosine_sim=cosine_sim, 
            movies_df=movies_df
        )
        
        # Create Prediction objects for recommended movies
        for rec_movie_title in recommended_movies:
            # Find movie ID from title
            rec_movie_id = movies_df[movies_df['title'] == rec_movie_title]['movieId'].values[0]
            est_rating = model.predict(uid, rec_movie_id).est
            predictions.append(Prediction(uid=uid, iid=rec_movie_title, est=est_rating, r_ui=None, details=None))
    
    # Identify positively rated movies in test set (rating >= 4)
    true_positive_movies = {movies_dict.get(mid) for uid, mid, true_r in testset if true_r >= 4}
    
    # Get set of recommended movies
    recommended_movies_set = {pred.iid for pred in predictions}
    
    # Calculate precision and recall
    true_positives = len(true_positive_movies.intersection(recommended_movies_set))
    precision = true_positives / len(recommended_movies_set) if len(recommended_movies_set) > 0 else 0
    recall = true_positives / len(true_positive_movies) if len(true_positive_movies) > 0 else 0
    
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    
    return precision, recall


def main():
    """
    Main function to run the movie recommendation system.
    """
    # Load data
    print("Loading data...")
    ratings_df, movies_df = load_data()
    
    # Display basic information about the data
    print("Ratings data preview:")
    print(ratings_df.head())
    print("\nMovies data preview:")
    print(movies_df.head())
    
    # Check for missing values
    print("\nMissing values in ratings data:")
    print(ratings_df.isnull().sum())
    
    # Train collaborative filtering model
    print("\nTraining collaborative filtering model...")
    model, testset = train_collaborative_filtering(ratings_df)
    
    # Evaluate collaborative filtering model
    print("\nEvaluating collaborative filtering model...")
    evaluate_collaborative_filtering(model, testset)
    
    # Create content-based recommender
    print("\nCreating content-based recommendation system...")
    cosine_sim, processed_movies_df = create_content_based_recommender(movies_df)
    
    # Evaluate hybrid system
    print("\nEvaluating hybrid recommendation system...")
    precision, recall = evaluate_hybrid_system(model, testset, cosine_sim, processed_movies_df)
    
    print(f"\nFinal Evaluation:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


if __name__ == "__main__":
    main()

