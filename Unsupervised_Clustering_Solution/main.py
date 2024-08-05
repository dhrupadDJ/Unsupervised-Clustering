from data_preprocessing.load_data import load_data
from data_preprocessing.preprocessdata import basic_statistics, plot_pairplot
from models.kmeans_clustering import train_kmeans, add_cluster_labels, get_cluster_centers
from evaluation.evaluation import calculate_wss, calculate_silhouette, plot_wss, plot_silhouette

# Load the data
file_path = 'mall_customers.csv'
df = load_data(file_path)

# Display basic statistics
print(basic_statistics(df))

# Plot pairplot for initial exploration
plot_pairplot(df, ['Age', 'Annual_Income', 'Spending_Score'])

# Train KMeans model with 5 clusters
kmeans_model = train_kmeans(df, n_clusters=5, features=['Annual_Income', 'Spending_Score'])

# Add cluster labels to the dataframe
df = add_cluster_labels(df, kmeans_model)

# Display cluster centers
print("Cluster Centers:\n", get_cluster_centers(kmeans_model))

# Evaluate clusters with WSS and Silhouette Score
wss_df = calculate_wss(df, range(3, 9), ['Annual_Income', 'Spending_Score'])
silhouette_df = calculate_silhouette(df, range(3, 9), ['Annual_Income', 'Spending_Score'])

# Plot WSS and Silhouette Score
plot_wss(wss_df)
plot_silhouette(silhouette_df)

# Further train with additional features
kmeans_model_full = train_kmeans(df, n_clusters=6, features=['Age', 'Annual_Income', 'Spending_Score'])

# Add new cluster labels
df = add_cluster_labels(df, kmeans_model_full)

# Evaluate clusters with WSS and Silhouette Score using all features
wss_df_full = calculate_wss(df, range(3, 9), ['Age', 'Annual_Income', 'Spending_Score'])
silhouette_df_full = calculate_silhouette(df, range(3, 9), ['Age', 'Annual_Income', 'Spending_Score'])

# Plot WSS and Silhouette Score for full feature set
plot_wss(wss_df_full)
plot_silhouette(silhouette_df_full)
