import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
import time


class ArticleRecommenderNN:

    def __init__(self, articles_df, epochs=50, batch_size=256):
        self.df = articles_df
        self.articles_content = self.df['cleaned_content'].tolist()

        # Initialize the TfidfVectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')

        # Create the TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.articles_content).toarray()

        # Neural network model
        self.model = Sequential()
        self.model.add(Dense(128, activation='relu', input_dim=self.tfidf_matrix.shape[1]))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.tfidf_matrix.shape[1], activation='linear'))  # Output the same size as input
        self.model.compile(optimizer='adam', loss='cosine_similarity')

        # Train the model on the articles content.
        # This assumes articles are similar to themselves, a form of autoencoder.
        self.model.fit(self.tfidf_matrix, self.tfidf_matrix, epochs=epochs, batch_size=batch_size, verbose=0)

    def recommend(self, given_articles_content, n_recommendations=5):

        start_time = time.time()

        # Transform each liked article into its TF-IDF representation
        given_articles_vectors = self.vectorizer.transform(given_articles_content).toarray()

        # Compute the average vector of all the liked articles
        avg_vector = np.mean(given_articles_vectors, axis=0).reshape(1, -1)

        # Use the model to get the recommended article vector
        recommended_vector = self.model.predict(avg_vector)

        # Find the most similar articles
        distances = np.linalg.norm(self.tfidf_matrix - recommended_vector, axis=1)
        recommended_indices = distances.argsort()[:n_recommendations + len(given_articles_content)]

        # Filter out the input articles from the recommendations
        final_indices = [i for i in recommended_indices if self.df['cleaned_content'][i] not in given_articles_content][
                        :n_recommendations]
        recommended_article_titles = [self.df.iloc[i]['title'] for i in final_indices]

        end_time = time.time()
        time_taken = end_time - start_time

        print(time_taken)
        return recommended_article_titles


