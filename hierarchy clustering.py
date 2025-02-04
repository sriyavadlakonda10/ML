import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from io import StringIO

st.title("Hierarchical Clustering with Streamlit")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset:")
    st.write(dataset.head())
    
    if 'Annual Income (k$)' in dataset.columns and 'Spending Score (1-100)' in dataset.columns:
        X = dataset[['Annual Income (k$)', 'Spending Score (1-100)']].values

        # Dendrogram
        st.write("### Dendrogram")
        fig, ax = plt.subplots()
        sch.dendrogram(sch.linkage(X, method='ward'))
        plt.title('Dendrogram')
        plt.xlabel('Customers')
        plt.ylabel('Euclidean distances')
        st.pyplot(fig)

        # Selecting number of clusters
        n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=5)

        # Hierarchical Clustering Model
        hc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
        y_hc = hc.fit_predict(X)
        
        # Visualizing Clusters
        st.write("### Clusters of Customers")
        fig, ax = plt.subplots()
        colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange', 'brown']
        
        for i in range(n_clusters):
            plt.scatter(X[y_hc == i, 0], X[y_hc == i, 1], s=100, c=colors[i], label=f'Cluster {i+1}')
        
        plt.title('Clusters of customers')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        st.pyplot(fig)
    else:
        st.write("Error: The uploaded CSV must contain 'Annual Income (k$)' and 'Spending Score (1-100)'")
else:
    st.write("Please upload a CSV file to proceed.")
