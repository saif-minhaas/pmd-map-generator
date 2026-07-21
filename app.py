import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import geopandas as gpd
from shapely.geometry import Point
from PIL import Image
import os

# Set Streamlit page configuration
st.set_page_config(page_title="PMD Weather Mapping System", layout="wide")

st.title("🗺️ PMD Meteorological Map Generator")
st.write("Upload station data to generate smooth, interpolated PMD reference maps.")

# --- IDW Interpolation Function ---
def arcgis_idw(x, y, z, grid_x, grid_y, power=2.0, k=5):
    """
    Local Inverse Distance Weighting (IDW) restricted to k-nearest neighbors.
    """
    tree = cKDTree(np.column_stack((x, y)))
    grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    distances, indices = tree.query(grid_pts, k=min(k, len(x)))
    
    # Avoid division by zero for exact match points
    distances = np.maximum(distances, 1e-12)
    
    weights = 1.0 / (distances ** power)
    sum_weights = np.sum(weights, axis=1)
    
    weighted_values = np.sum(weights * z[indices], axis=1) / sum_weights
    return weighted_values.reshape(grid_x.shape)

# --- Sidebar Controls ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Station Spreadsheet (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read uploaded file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.sidebar.header("2. Column Selection")
    lat_col = st.sidebar.selectbox("Latitude Column", df.columns, index=0)
    lon_col = st.sidebar.selectbox("Longitude Column", df.columns, index=1)
    value_col = st.sidebar.selectbox("Value/Data Column", df.columns, index=2)

    # --- Safe Data Cleaning ---
    df_clean = df.copy()
    df_clean[lat_col] = pd.to_numeric(df_clean[lat_col], errors='coerce')
    df_clean[lon_col] = pd.to_numeric(df_clean[lon_col], errors='coerce')
    df_clean[value_col] = pd.to_numeric(df_clean[value_col], errors='coerce')

    # Drop any row containing missing lat, lon, or value
    df_clean = df_clean.dropna(subset=[lat_col, lon_col, value_col]).copy()

    st.subheader("Data Preview")
    st.dataframe(df_clean.head())

    # --- Generate Map Action ---
    if st.button("Generate Final PMD Reference Map", key="btn_generate_map"):
        with st.spinner("Processing spatial interpolation and rendering map..."):
            
            x = df_clean[lon_col].values
            y = df_clean[lat_col].values
            z = df_clean[value_col].values

            # Create bounding grid for Pakistan
            min_lon, max_lon = 60.0, 78.0
            min_lat, max_lat = 23.5, 37.5
            
            grid_x, grid_y = np.mgrid[min_lon:max_lon:500j, min_lat:max_lat:500j]

            # Compute IDW grid
            grid_z = arcgis_idw(x, y, z, grid_x, grid_y, power=2.0, k=5)

            # Setup Matplotlib Figure
            fig, ax = plt.subplots(figsize=(10, 10), facecolor='#FDFDF0')
            ax.set_facecolor('#FDFDF0')

            # Contour plot
            levels = np.linspace(np.nanmin(z), np.nanmax(z), 15)
            contour = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap='Spectral_r', extend='both')

            # Overlay station points
            ax.scatter(x, y, color='black', s=15, zorder=5, label="PMD Stations")

            # Load shapefile if available
            shp_path = "Province.shp"
            if os.path.exists(shp_path):
                gdf = gpd.read_file(shp_path)
                gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.2, zorder=4)

            # Map Aesthetics
            ax.set_xlim(min_lon, max_lon)
            ax.set_ylim(min_lat, max_lat)
            ax.set_title("Pakistan Meteorological Department - Spatial Analysis", fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            
            plt.colorbar(contour, ax=ax, shrink=0.7, label="Value Scale")
            ax.legend(loc="upper left")

            # Display plot in Streamlit
            st.pyplot(fig)

else:
    st.info("👆 Please upload a station data file from the sidebar to begin.")
