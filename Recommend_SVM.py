from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from scipy.sparse import vstack
import numpy as np
import time


class ArticleRecommenderSVM:

    def __init__(self, articles_df):
        self.df = articles_df
        self.articles_content = self.df['cleaned_content'].tolist()

        # Initialize the TfidfVectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')

        # Create the TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.articles_content)

        # Create an SVC object
        self.svm = SVC(kernel='linear', probability=True)

        # For this example, we won't fit the SVM here; it will be fitted for every recommendation request

    def recommend(self, given_articles_content, n_recommendations=5):

        start_time = time.time()

        # Transform each liked article into its TF-IDF representation
        given_articles_vectors = self.vectorizer.transform(given_articles_content)

        # Compute the average vector of all the liked articles
        avg_vector = np.mean(given_articles_vectors, axis=0)  # This remains sparse

        # We create a simple training set: positive examples are the liked articles, negative examples are randomly
        # sampled
        x_train = given_articles_vectors
        y_train = [1] * x_train.shape[0]

        # Sample other articles for negative examples
        np.random.seed(42)  # For reproducibility
        negative_samples = np.random.choice(self.articles_content, len(given_articles_content))
        negative_samples_vectors = self.vectorizer.transform(negative_samples)
        x_train = vstack([x_train, negative_samples_vectors])  # Stacking sparse matrices
        y_train.extend([0] * len(given_articles_content))

        # Fit the SVM
        self.svm.fit(x_train, y_train)

        # Get the decision function values for all articles
        distances = self.svm.decision_function(self.tfidf_matrix)

        # Get the indices of the articles sorted by their relevance (higher decision function value is more relevant)
        recommended_indices = distances.argsort()[::-1]

        # Filter out the input articles from the recommendations
        recommended_indices = [i for i in recommended_indices if
                               self.df['cleaned_content'].iloc[i] not in given_articles_content][:n_recommendations]
        recommended_article_titles = [self.df.iloc[i]['title'] for i in recommended_indices]

        end_time = time.time()
        time_taken = end_time - start_time

        print(time_taken)
        return recommended_article_titles

