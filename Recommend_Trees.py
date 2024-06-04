from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import time


class ArticleRecommenderDT:

    def __init__(self, articles_df, n_clusters=10):
        self.df = articles_df
        self.articles_content = self.df['cleaned_content'].tolist()

        # Initialize the TfidfVectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')

        # Create the TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.articles_content)

        # Cluster the articles
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.df['cluster'] = self.kmeans.fit_predict(self.tfidf_matrix)

    def recommend(self, given_articles_content, n_recommendations=5):
        start_time = time.time()

        # Transform each liked article into its TF-IDF representation
        given_articles_vectors = self.vectorizer.transform(given_articles_content)

        # For each liked article, train a Random Forest model to predict its cluster
        liked_clusters = []
        for idx, vec in enumerate(given_articles_vectors):
            x = self.tfidf_matrix
            y = (self.df['cluster'] ==
                 self.df.loc[self.df['cleaned_content'] == given_articles_content[idx], 'cluster'].iloc[0]).astype(int)
            rf = RandomForestClassifier()
            rf.fit(x, y)
            liked_clusters.append(rf.predict(vec)[0])

        # Recommend articles from the same clusters
        recommended_indices = self.df[self.df['cluster'].isin(liked_clusters)].index.tolist()
        recommended_article_titles = [self.df.iloc[i]['title'] for i in recommended_indices if
                                      self.df['cleaned_content'][i] not in given_articles_content][:n_recommendations]

        end_time = time.time()
        time_taken = end_time - start_time

        print(time_taken)
        return recommended_article_titles


