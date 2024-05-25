import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st

data = pd.read_csv(r'D:\Internship tasks\Task1and2\train.csv', encoding='latin1')

# data.head()
# data.isnull().sum()

x_train = data.drop(columns=["target"]).values

# x_train

def preditCluster(data_point):
    cluster = kmeans.predict(data_point.reshape(1, -1))[0]
    centroid = kmeans.cluste_center_[cluster]
    distance = np.linalg.norm(data_point - centroid)
    return cluster, distance


# Train K-means clustering model
kmeans = KMeans(n_clusters=5, random_state=42)  # Number of clusters chosen arbitrarily
kmeans.fit(x_train)

test_data = pd.read_csv(r'D:\Internship tasks\Task1and2\test.csv', encoding='latin1')
# test_data.head()

x_test = test_data.values

# x_test

test_data["predicted_cluster"] = kmeans.predict(x_test)
# test_data




#! Streamlit App
st.title('Cluster Prediction App')

# Sidebar for input data
st.sidebar.header('Input Data')
input_data = st.sidebar.file_uploader('Upload CSV', type=['csv', 'xlsx'])

# Display uploaded data
if input_data is not None:
    df = pd.read_csv(input_data)  # Assuming input data is in CSV format
    st.write(df)

    # Predict clusters
    predicted_clusters = kmeans.predict(df)  # Assuming df contains input data in the same format as training data

    # Display predicted clusters
    st.write('Predicted Clusters:')
    st.write(predicted_clusters)
