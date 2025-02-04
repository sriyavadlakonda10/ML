# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Title of the Streamlit app
st.title('K-Means Clustering on Mall Customers Dataset')

# Upload the dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a CSV file.")
    st.stop()

# Select the features for clustering
st.sidebar.header("Select Features for Clustering")
feature_columns = dataset.columns.tolist()
selected_features = st.sidebar.multiselect("Choose features", feature_columns, default=feature_columns[3:5])

# Extract the selected features
X = dataset[selected_features].values

# Using the elbow method to find the optimal number of clusters
st.header("Elbow Method to Find Optimal Number of Clusters")
from sklearn.cluster import KMeans

wcss = []
max_clusters = st.sidebar.slider("Maximum number of clusters to test", 2, 15, 10)

for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the elbow method graph
fig, ax = plt.subplots()
ax.plot(range(1, max_clusters + 1), wcss, marker='o')
ax.set_title('The Elbow Method')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('WCSS')
st.pyplot(fig)

# Select the optimal number of clusters
optimal_clusters = st.sidebar.number_input("Enter the optimal number of clusters", min_value=2, max_value=max_clusters, value=5)

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
st.header("Visualizing the Clusters")
fig, ax = plt.subplots()
for i in range(optimal_clusters):
    ax.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, label=f'Cluster {i+1}')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
ax.set_title('Clusters of customers')
ax.set_xlabel(selected_features[0])
ax.set_ylabel(selected_features[1])
ax.legend()
st.pyplot(fig)