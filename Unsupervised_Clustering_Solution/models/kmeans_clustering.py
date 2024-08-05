from sklearn.cluster import KMeans

def train_kmeans(df, n_clusters, features):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++').fit(df[features])
    return kmeans

def add_cluster_labels(df, kmeans_model):
    df['Cluster'] = kmeans_model.labels_
    return df

def get_cluster_centers(kmeans_model):
    return kmeans_model.cluster_centers_

def get_labels(kmeans_model):
    return kmeans_model.labels_
