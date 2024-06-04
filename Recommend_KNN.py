from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import time


class ArticleRecommenderKNN:

    def __init__(self, articles_df):
        self.df = articles_df
        self.articles_content = self.df['cleaned_content'].tolist()

        # Initialize the TfidfVectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')

        # Create the TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.articles_content)

        # Create a NearestNeighbors object
        self.knn = NearestNeighbors(metric='euclidean', algorithm='brute')

        # Fit the model to your data
        self.knn.fit(self.tfidf_matrix)

    def recommend(self, given_articles_content, n_recommendations=5):
        start_time = time.time()
        # Transform each liked article into its TF-IDF representation
        given_articles_vectors = self.vectorizer.transform(given_articles_content)

        # Compute the average vector of all the liked articles
        avg_vector = np.mean(given_articles_vectors.toarray(), axis=0)

        # Find the nearest neighbors to the average vector
        distances, indices = self.knn.kneighbors([avg_vector],
                                                 n_neighbors=n_recommendations + len(given_articles_content))

        # Filter out the input articles from the recommendations
        recommended_indices = [i for i in indices[0] if self.df['cleaned_content'][i] not in given_articles_content][
                              :n_recommendations]
        recommended_article_titles = [self.df.iloc[i]['title'] for i in recommended_indices]

        end_time = time.time()
        time_taken = end_time - start_time

        print(time_taken)
        return recommended_article_titles
