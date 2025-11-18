import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os  # <-- for checking dataset existence

# ---------------------- PAGE SETUP ---------------------- #
st.set_page_config(page_title="Student Cluster Predictor", page_icon="🎯")

st.title("🎯 Student Cluster Prediction (K-Means)")
st.markdown("Enter the student details to predict the cluster they belong to.")

# ---------------------- LOAD DATA ---------------------- #
@st.cache_data
def load_data(path):
    return pd.read_excel(path)

# ---------------------- SAFETY CHECK FOR DATASET ---------------------- #
dataset_path = "students_performance_dataset.xlsx"

if not os.path.exists(dataset_path):
    st.error(f"Dataset not found at {dataset_path}")
    st.stop()

df = load_data(dataset_path)

# ---------------------- FEATURE SELECTION ---------------------- #
FEATURES = [
    'Study_Hours_per_Week',
    'Attendance_Percentage',
    'Previous_Sem_Score',
    'Library_Usage_per_Week',
    'Sleep_Hours',
    'Motivation_Level',
    'Test_Anxiety_Level',
    'Peer_Influence'
]

# Keep only features that exist in the dataset
FEATURES = [f for f in FEATURES if f in df.columns]

df_feat = df[FEATURES].dropna()
X = df_feat.values.astype(float)

# ---------------------- SCALE FEATURES ---------------------- #
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------- TRAIN K-MEANS ---------------------- #
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# ---------------------- USER INPUT ---------------------- #
st.subheader("Enter Student Details")

user_vals = {}
for f in FEATURES:
    col = df[f].dropna()
    minv = float(col.min())
    maxv = float(col.max())
    meanv = float(col.mean())

    user_vals[f] = st.slider(
        f.replace("_", " "),
        min_value=minv,
        max_value=maxv,
        value=meanv,
        step=1.0
    )

# ---------------------- PREDICTION ---------------------- #
if st.button("Predict Cluster"):
    user_array = np.array(list(user_vals.values())).reshape(1, -1)
    user_scaled = scaler.transform(user_array)
    cluster = int(kmeans.predict(user_scaled)[0])

    st.success(f"### 🎯 Predicted Cluster: **{cluster}**")

    # Meaning based on Previous Sem Score
    df_feat_clustered = df_feat.copy()
    df_feat_clustered['Cluster'] = kmeans.predict(X_scaled)
    cluster_means = df_feat_clustered.groupby('Cluster')['Previous_Sem_Score'].mean().to_dict()

    if cluster_means[cluster] >= 80:
        meaning = "High-performing student"
    elif cluster_means[cluster] >= 65:
        meaning = "Moderate performer"
    else:
        meaning = "Low-performing / At-risk group"

    st.info(f"### Interpretation: **{meaning}**")

