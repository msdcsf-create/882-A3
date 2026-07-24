import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# SETUP NOTES
# To run this in VS, I need to do following in terminal:
# 1. Activate my virtual environment: .\venv\Scripts\Activate.ps1
# 2. Install requirements: pip install streamlit matplotlib pandas scikit-learn
# 3. Run the command: streamlit run appPB.py

# Pagbe configuration
st.set_page_config(page_title='Vancouver Business Clustering', layout='wide')


# @st.cache_data is sued so Streamlit doesn't reload CSV every time I move a slider.
@st.cache_data
def load_data():
    # app expects the clean data from Part A
    path = os.path.join('data', 'business_clean.csv')
    if not os.path.exists(path):
        st.error(f'Oops! I cannot find {path}. Did I export it from the notebook?')
        return None
    return pd.read_csv(path)

st.title('A03 Vancouver Business License Clustering')
st.sidebar.header('Control Panel')

df = load_data()

if df is not None:
    # 1. SIDEBAR Let user choose K
    # I chose a range of 2 to 15 based on my Elbow Method results from Part A.
    k_value = st.sidebar.slider('How many clusters should we find?', min_value=2, max_value=15, value=5)
    
    # 2. SPATIAL CLUSTERING: 
    # using Lat/Lon to group businesses by their actual physical location.
    coords = df[['latitude', 'longitude']]
    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(coords)
    
    # 3. DISPLAY RESULTS: Splitting the screen into two columns for a clean look.
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f'Map: Where the {k_value} Clusters Are')
        fig_map, ax_map = plt.subplots(figsize=(10, 8))
        # using the 'tab10' color scheme so each cluster has a distinct color
        ax_map.scatter(
            df['longitude'], 
            df['latitude'], 
            c=df['cluster'], 
            cmap='tab10', 
            s=2, 
            alpha=0.6
        )
        ax_map.set_title('Geographic Business Hubs')
        ax_map.set_xlabel('Longitude')
        ax_map.set_ylabel('Latitude')
        st.pyplot(fig_map)

    with col2:
        st.subheader('Cluster Sizes')
        # a quick bar chart to see if one cluster is way bigger than the others
        counts = df['cluster'].value_counts().sort_index()
        st.bar_chart(counts)
        
        # A big counter at the bottom to show how many total records are loaded
        st.metric("Total Licenses Analyzed", len(df))

    # 4. Feature clustering (PCA): 
    # This part looks at the characteristics (staff size, fees, etc.) rather than location.
    st.divider()
    st.subheader('Feature-Based Visualization (PCA)')
    
    features = ['numberofemployees', 'feepaid', 'duration_days', 'licencerevisionnumber']
    
    if all(col in df.columns for col in features):
        # Fill any missing values with 0 so the math doesn't break
        X = df[features].fillna(0)
        
        # SCALE: I scale the data so 'Number of Employees' doesn't overpower 'Fee Paid'
        X_scaled = StandardScaler().fit_transform(X)
        
        # PCA: Squash our 4 dimensions down to 2 so we can plot them on a flat graph
        pca = PCA(n_components=2)
        pca_points = pca.fit_transform(X_scaled)
        
        fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
        # Color these points by the spatial clusters to see if neighborhoods share business traits
        ax_pca.scatter(pca_points[:, 0], pca_points[:, 1], c=df['cluster'], cmap='tab10', s=5)
        ax_pca.set_xlabel('PC1 (Main Variation)')
        ax_pca.set_ylabel('PC2 (Second Variation)')
        st.pyplot(fig_pca)
else:
    st.info('Waiting for data... Please ensure business_clean.csv is in the data/ folder.')