import pandas as pd
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def calculate_wss(df, range_n_clusters, features):
    WCSS = []
    for n in range_n_clusters:
        kmeans = KMeans(n_clusters=n).fit(df[features])
        wcss_score = kmeans.inertia_
        WCSS.append(wcss_score)
    return pd.DataFrame({'cluster': range_n_clusters, 'WSS_Score': WCSS})

def calculate_silhouette(df, range_n_clusters, features):
    silhouette_scores = []
    for n in range_n_clusters:
        kmeans = KMeans(n_clusters=n).fit(df[features])
        silhouette_scores.append(silhouette_score(df[features], kmeans.labels_))
    return pd.DataFrame({'cluster': range_n_clusters, 'Silhouette_Score': silhouette_scores})

def plot_wss(wss_df):
    wss_df.plot(x='cluster', y='WSS_Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within-Cluster Sum of Squares (WSS)')
    plt.title('Elbow Method')
    plt.show()

def plot_silhouette(silhouette_df):
    silhouette_df.plot(x='cluster', y='Silhouette_Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.show()
