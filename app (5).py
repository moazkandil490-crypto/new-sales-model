import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Sales Transaction Clustering App",
    page_icon="📊",
    layout="centered"
)

# Load the saved model and scaler
@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.joblib')
    cluster_model = joblib.load('cluster_model.joblib')
    return scaler, cluster_model

try:
    scaler, cluster_model = load_assets()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")

# Define cluster profiles
cluster_descriptions = {
    0: "**Cluster 0 (Low-value segment):** Transactions characterized by smaller quantities, lower unit prices, and overall low sales values.",
    1: "**Cluster 1 (High-value segment):** Transactions characterized by larger quantities, premium unit prices, and high sales values.",
    2: "**Cluster 2 (Average-value segment):** Transactions characterized by lower quantities but premium unit prices, producing mid-tier sales values."
}

# App layout
st.title("📊 Sales Transaction Segment Predictor")
st.write("Input a transaction's parameters below to determine its hierarchical customer segment classification.")

# Input fields
col1, col2, col3 = st.columns(3)
with col1:
    quantity = st.number_input("Quantity Ordered", min_value=1, value=30, step=1)
with col2:
    price = st.number_input("Price Each ($)", min_value=0.0, value=95.0, step=0.5)
with col3:
    sales = st.number_input("Total Sales ($)", min_value=0.0, value=2800.0, step=10.0)

# Prediction trigger
if st.button("Classify Segment", type="primary"):
    # 1. Structure input into a DataFrame
    input_data = pd.DataFrame([[quantity, price, sales]], columns=['QUANTITYORDERED', 'PRICEEACH', 'SALES'])
    
    # 2. Scale features using the loaded scaler
    input_scaled = scaler.transform(input_data)
    
    # 3. Predict cluster assignment (Mapping to nearest centroid in scaled space)
    centroids = [
        [-0.114, -1.218, -0.669], 
        [ 0.449,  0.742,  1.326],
        [-0.743,  0.528, -0.217]
    ]
    
    distances = [np.linalg.norm(input_scaled[0] - c) for c in centroids]
    predicted_cluster = np.argmin(distances)
    
    # Display Results
    st.success(f"### Prediction: Segment Cluster {predicted_cluster}")
    st.write(cluster_descriptions[predicted_cluster])
