import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from shapely.geometry import MultiPolygon, Polygon
from PIL import Image
import os

# Geopandas conditional import to maintain runtime flexibility
gpd = None
try:
    import geopandas as gpd
except ImportError:
    pass

# Page Setup
st.set_page_config(page_title="PMD Weather Mapping System", layout="wide")
st.title("🗺️ PMD Meteorological Map Generator")

# Sidebar - File Uploads
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Station Spreadsheet Data (CSV/Excel)", type=["csv", "xlsx"])

@st.cache_data
def load_base_map():
    if 'gpd' in globals() and gpd is not None and os.path.exists("Province.shp"):
        try:
            return gpd.read_file("Province.shp")
        except Exception:
            return gpd.read_file("Province.shp", engine='fiona')
    return None

gdf_province = load_base_map()
default_logo_path = "logo.png"

# Clipping function for administrative boundary locking
def get_clip_patch(gdf, ax):
    unified_geom = gdf.unary_union
    if isinstance(unified_geom, Polygon):
        polygons = [unified_geom]
    elif isinstance(unified_geom, MultiPolygon):
        polygons = list(unified_geom.geoms)
    else:
        return None

    paths = []
    for poly in polygons:
        exterior = np.array(poly.exterior.coords)
        paths.append(Path(exterior))
        for interior in poly.interiors:
            paths.append(Path(np.array(interior.coords)))
            
    if paths:
        combined_path = Path.make_compound_path(*paths)
        patch = PathPatch(combined_path, transform=ax.transData, facecolor='none', edgecolor='none')
        return patch
    return None

# ARCGIS IDW ENGINE OPTIMIZED WITH EXACTLY 5 NEIGHBORS
def arcgis_idw(x, y, z, grid_x, grid_y, power=2.0, epsilon=1e-6):
    stations = np.vstack((x, y)).T
    grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
    
    tree = cKDTree(stations)
    k_neighbors = min(5, len(x))
    distances, indices = tree.query(grid_points, k=k_neighbors)
    
    distances = np.maximum(distances, epsilon)
    weights = 1.0 / (distances ** power)
    
    if k_neighbors == 1:
        weighted_values = z[indices]
    else:
        sum_weights = np.sum(weights, axis=1)
        weighted_values = np.sum(weights * z[indices], axis=1) / sum_weights
        
    return weighted_values.reshape(grid_x.shape)

# Main Application Logic
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    columns = df.columns.tolist()
    
    lat_col = st.sidebar.selectbox("Select Latitude Column:", [c for c in columns if 'lat' in str(c).lower() or 'y' in str(c).lower()] + columns)
    lon_col = st.sidebar.selectbox("Select Longitude Column:", [c for c in columns if 'long' in str(c).lower() or 'x' in str(c).lower()] + columns)
    station_col = st.sidebar.selectbox("Select Station Name Column:", [c for c in columns if 'station' in str(c).lower() or 'name' in str(c).lower()] + columns)
    value_col = st.sidebar.selectbox("Select Value Column:", [c for c in columns if 'dep' in str(c).lower() or 'temp' in str(c).lower() or 'rain' in str(c).lower() or 'prec' in str(c).lower()] + columns)

    prov_name_col = None
    if gdf_province is not None:
        possible_cols = [c for c in gdf_province.columns if any(sub in str(c).lower() for sub in ['name', 'prov', 'dist', 'admin'])]
        if possible_cols:
            prov_name_col = st.sidebar.selectbox("Select Region Label Column (Shapefile):", possible_cols)

    st.sidebar.header("2. Map Options")
    show_labels = st.sidebar.checkbox("Show District / Station Labels", value=True)
    
    map_title_input = st.sidebar.text_input("Map Title:", value="Rainfall July 2026")

    map_theme = st.sidebar.selectbox("Select Color Palette Preset:", [
        "For Rainfall (PMD Style)",
        "For Temperature (Heat Map)", 
        "For Departures"
    ])

    rain_scale = None
    if map_theme == "For Rainfall (PMD Style)":
        rain_scale = st.sidebar.selectbox("Select Climate Season / Rainfall Scale:", [
            "For Normal Season (100 mm Max)",
            "For Monsoon Season (400 mm Max)",
            "For Daily Rainfall (Custom PMD Scale)"
        ])
    
    # Safe Unified Data Cleaning
    df_clean = df.copy()
    df_clean[lat_col] = pd.to_numeric(df_clean[lat_col], errors='coerce')
    df_clean[lon_col] = pd.to_numeric(df_clean[lon_col], errors='coerce')
    df_clean[value_col] = pd.to_numeric(df_clean[value_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[lat_col, lon_col, value_col]).copy()

    if st.button("Generate Final PMD Reference Map", key="btn_generate_map"):
        with st.spinner("Rendering 5-neighbor restricted smooth contours..."):
            
            fig = plt.figure(figsize=(11, 10), facecolor='#FDFDF0')
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.75], facecolor='#FDFDF0')

            x = df_clean[lon_col].values
            y = df_clean[lat_col].values
            z = df_clean[value_col].values

            if gdf_province is not None:
                minx, miny, maxx, maxy = gdf_province.total_bounds
                minx -= 0.5; maxx += 0.5; miny -= 0.5; maxy += 0.5
            else:
                minx, maxx = 60.0, 81.0
                miny, maxy = 23.0, 38.0

            # CONFIGURING EXACT SCALES
            if map_theme == "For Rainfall (PMD Style)":
                if "Normal" in rain_scale:
                    pmd_colors = ['#eaeaea', '#b0b0b0', '#7da2cc', '#3366cc', '#f3a683', '#e15f41', '#c44569', '#571822']
                    bounds = [-999, 0.1, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 9999]
                    labels = ['< 0.1', '0.1 - 9.9', '10 - 19.9', '20 - 39.9', '40 - 59.9', '60 - 79.9', '80 - 99.9', '> 100']
                elif "Monsoon" in rain_scale:
                    pmd_colors = ['#a6a6a6', '#7f92b1', '#1f5ca3', '#003399', '#3cb371', '#009933', '#006600', '#cccc00', '#d35400', '#e06666', '#cc0000', '#990000']
                    bounds = [-999, 0.1, 30, 60, 90, 120, 150, 200, 250, 300, 350, 400, 9999]
                    labels = ['< 0.1', '0.1 - 29.9', '30 - 59.9', '60 - 89.9', '90 - 119.9', '120 - 149.9', '150 - 199.9', '200 - 249.9', '250 - 299.9', '300 - 349.9', '350 - 399.9', '> 400']
                else:  # For Daily Rainfall
                    pmd_colors = ['#ffffff', '#eaeaea', '#4ce600', '#4f81bd', '#1f497d', '#d99694', '#df65b0', '#e31a1c', '#7a0016']
                    bounds = [-0.01, 0.0, 0.1, 7.5, 15.5, 35.5, 65.5, 124.5, 244.5, 9999]
                    labels = ['0', '< 0.1', '0.1 - 7.5', '7.6 - 15.5', '15.6 - 35.5', '35.6 - 65.5', '65.6 - 124.5', '124.6 - 244.5', '> 244.5']
                legend_title = "Rainfall (mm)"
            
            elif map_theme == "For Temperature (Heat Map)":
                pmd_colors = ['#41719c', '#7da2cc', '#df8d93', '#d55861', '#e12230', '#a31c24']
                bounds = [23, 27.9, 31.9, 35.9, 39.9, 43.9, 999]
                labels = ['23 - 27', '28 - 31', '32 - 35', '36 - 39', '40 - 43', '44 - 47']
                legend_title = "Temperature (°C)"
                
            else:
                pmd_colors = ['#08306b', '#2171b5', '#6baed6', '#bdd7e7', '#f0f0f0', '#fcae91', '#fb6a4a', '#cb181d', '#67000d']
                bounds = [-999, -6.5, -4.5, -3.0, -1.5, 1.5, 3.0, 4.5, 6.5, 999]
                labels = ['<-6.5', '-6.5 - -4.5', '-4.5 - -3.0', '-3.0 - -1.5', '-1.5 - 1.5', '1.5 - 3.0', '3.0 - 4.5', '4.5 - 6.5', '>6.5']
                legend_title = "Deviation (°C)"
            
            custom_cmap = ListedColormap(pmd_colors)
            norm = BoundaryNorm(bounds, custom_cmap.N)

            # High-Resolution Matrix Mesh
            grid_x, grid_y = np.mgrid[minx:maxx:500j, miny:maxy:500j]
            
            # RUN COMPLIANT LOCALIZED 5-NEIGHBOR IDW
            grid_z = arcgis_idw(x, y, z, grid_x, grid_y, power=2.0)

            im = ax.imshow(grid_z.T, extent=(minx, maxx, miny, maxy), origin='lower', cmap=custom_cmap, norm=norm, alpha=0.95)

            if gdf_province is not None:
                clip_patch = get_clip_patch(gdf_province, ax)
                if clip_patch is not None:
                    ax.add_patch(clip_patch)
                    im.set_clip_path(clip_patch)

            if gdf_province is not None:
                gdf_province.plot(ax=ax, facecolor='none', edgecolor='#222222', linewidth=0.9, zorder=3)
                ax.set_xlim(minx, maxx)
                ax.set_ylim(miny, maxy)

            if gdf_province is not None and prov_name_col is not None and show_labels:
                for idx, row in gdf_province.iterrows():
                    centroid = row.geometry.centroid
                    if minx <= centroid.x <= maxx and miny <= centroid.y <= maxy:
                        ax.text(centroid.x, centroid.y, str(row[prov_name_col]), 
                                fontsize=7, color='#444444', weight='bold', 
                                ha='center', va='center', zorder=4,
                                bbox=dict(facecolor='#ffffff', alpha=0.4, edgecolor='none', pad=1))

            ax.scatter(x, y, color='black', s=8, zorder=5)
            
            if show_labels:
                for i, txt in enumerate(df_clean[station_col]):
                    ax.annotate(str(txt), (x[i], y[i]), fontsize=6, fontweight='bold', color='black',
                                xytext=(3, 3), textcoords='offset points', zorder=6)

            # Legend Layout
            patches = [mpatches.Patch(color=pmd_colors[i], label=labels[i]) for i in range(len(pmd_colors))]
            if map_theme == "For Rainfall (PMD Style)":
                patches.reverse()
                
            leg = ax.legend(handles=patches, loc='center right', bbox_to_anchor=(0.98, 0.45), 
                            title=legend_title, frameon=True, facecolor='white', edgecolor='grey', fontsize=8)
            leg.get_title().set_fontsize('9')
            leg.get_title().set_weight('bold')

            if os.path.exists(default_logo_path):
                logo_img = Image.open(default_logo_path)
                fig.figimage(logo_img, xo=135, yo=1180, alpha=1.0, zorder=10)

            ax.text(0.95, 0.94, 'N\n▲', transform=ax.transAxes, fontsize=11, weight='bold', ha='center', va='center')
            ax.text(0.91, 0.91, 'W', transform=ax.transAxes, fontsize=8, weight='bold')
            ax.text(0.99, 0.91, 'E', transform=ax.transAxes, fontsize=8, weight='bold')
            ax.text(0.95, 0.88, 'S', transform=ax.transAxes, fontsize=8, weight='bold')

            scale_length_deg = 400 / 111.0 
            start_x, start_y = minx + 10.5, miny + 0.4  
            ax.plot([start_x, start_x + scale_length_deg], [start_y, start_y], color='black', linewidth=3, zorder=5)
            ax.plot([start_x, start_x + scale_length_deg/2], [start_y, start_y], color='white', linewidth=1.5, zorder=6)
            ax.text(start_x, start_y + 0.15, '0', fontsize=7, ha='center')
            ax.text(start_x + scale_length_deg/2, start_y + 0.15, '200', fontsize=7, ha='center')
            ax.text(start_x + scale_length_deg, start_y + 0.15, '400 Kilometers', fontsize=7, ha='center')

            ax.set_title(map_title_input, fontsize=14, fontweight='bold', fontfamily='serif', pad=35)
            
            ax.text(0.02, 0.03, "Source: NWFC, PMD", transform=ax.transAxes, 
                    color='#3182bd', fontsize=9, fontweight='bold', ha='left', va='bottom', zorder=10)

            ax.set_xticklabels([f"{val}°E" for val in ax.get_xticks()])
            ax.set_yticklabels([f"{val}°N" for val in ax.get_yticks()])
            ax.tick_params(labelsize=9)

            st.pyplot(fig)

    st.subheader("Sheet Preview")
    st.dataframe(df_clean.head(5), use_container_width=True)
