#!/usr/bin/env python
# coding: utf-8

# ### section1

# In[3]:


# Install required libraries (run once if needed)
#!pip install rasterio geopandas matplotlib-scalebar tensorflow fiona

import numpy as np
import tensorflow as tf
import os
import gc
import multiprocessing as mp
from functools import partial
import time
from tensorflow.keras import layers, models
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from scipy.ndimage import sobel, gaussian_filter
from scipy.interpolate import griddata
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib_scalebar.scalebar import ScaleBar
import geopandas as gpd
from shapely.geometry import Point
import fiona
import zipfile
import tempfile
import shutil

# Check device and enable GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is active:", physical_devices)
else:
    print("GPU not found, running on CPU.")

# Functions
def convert_kmz_to_shapefile(kmz_path, output_shp_path):
    with zipfile.ZipFile(kmz_path, 'r') as kmz:
        kmz.extractall(tempfile.gettempdir())
        kml_file = None
        for file in kmz.namelist():
            if file.endswith('.kml'):
                kml_file = os.path.join(tempfile.gettempdir(), file)
                break
        if not kml_file:
            shutil.rmtree(tempfile.gettempdir(), ignore_errors=True)
            raise FileNotFoundError("No KML file found inside KMZ.")
    with fiona.open(kml_file, 'r') as kml:
        gdf = gpd.GeoDataFrame.from_features([feature for feature in kml])
    gdf.to_file(output_shp_path)
    shutil.rmtree(tempfile.gettempdir(), ignore_errors=True)
    return output_shp_path

def align_raster(input_path, transform_ref, crs_ref, shape_ref):
    with rasterio.open(input_path) as src:
        data = src.read(1)
        data_aligned = np.zeros(shape_ref, dtype=data.dtype)
        reproject(source=data, destination=data_aligned, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform_ref, dst_crs=crs_ref, resampling=Resampling.cubic)
        return data_aligned

def calculate_slope_aspect(dem):
    dx = sobel(dem, axis=1)
    dy = sobel(dem, axis=0)
    slope = np.sqrt(dx**2 + dy**2)
    aspect = np.arctan2(dy, dx)
    return np.nan_to_num(slope, nan=0.0), np.nan_to_num(aspect, nan=0.0)

def calculate_tri(dem, window_size=3):
    tri = np.zeros_like(dem, dtype=np.float32)
    h, w = dem.shape
    half_window = window_size // 2
    for i in range(half_window, h - half_window):
        for j in range(half_window, w - half_window):
            window = dem[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
            if not np.any(np.isnan(window)):
                tri[i, j] = np.abs(window - window[half_window, half_window]).mean()
    return np.nan_to_num(tri, nan=0.0)

def create_patches(data, patch_size=8, stride=32):
    patches = []
    h, w, c = data.shape
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = data[i:i+patch_size, j:j+patch_size]
            # Keep patch if less than 50% of it is NaN or 0
            if np.count_nonzero(patch == 0) < (patch_size * patch_size * c * 0.5):
                patches.append(patch)
    return np.array(patches)

def fill_nan_with_interpolation(data, x, y, method='nearest'):
    valid_mask = ~np.isnan(data)
    valid_points = np.column_stack((x[valid_mask], y[valid_mask]))
    valid_values = data[valid_mask]
    invalid_points = np.column_stack((x[~valid_mask], y[~valid_mask]))
    if len(valid_points) > 0 and len(invalid_points) > 0:
        filled_values = griddata(valid_points, valid_values, invalid_points, method=method, fill_value=0)
        data[~valid_mask] = filled_values
    return data

# Log start time
start_time = time.time()
print(f"Execution started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

# Create output directory
output_dir = 'outputs/section_1'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created/checked: {output_dir}")

# Load data
print("Loading data...")
try:
    with rasterio.open('clipped_dinsar.tif') as src_dinsar:
        d_los = src_dinsar.read(1)
        transform_ref = src_dinsar.transform
        crs_ref = src_dinsar.crs
        shape_ref = (src_dinsar.height, src_dinsar.width)
        lons, lats = np.meshgrid(np.linspace(src_dinsar.bounds.left, src_dinsar.bounds.right, src_dinsar.width),
                                 np.linspace(src_dinsar.bounds.bottom, src_dinsar.bounds.top, src_dinsar.height))
    print("Loaded clipped_dinsar.tif, shape:", d_los.shape)
except Exception as e:
    print(f"Error loading clipped_dinsar.tif: {e}")

try:
    psinsar_df = pd.read_csv('alaska_ps.csv')
    psinsar_lons = psinsar_df['longitude'].values
    psinsar_lats = psinsar_df['latitude'].values
    psinsar_deff = psinsar_df['deff'].values
    points = np.stack([psinsar_lons, psinsar_lats], axis=1)
    dz_psinsar = griddata(points, psinsar_deff, (lons, lats), method='cubic')
    dz_psinsar = np.where(np.isnan(dz_psinsar), 0, dz_psinsar)
    print("Loaded alaska_ps.csv, dz_psinsar shape:", dz_psinsar.shape)
except Exception as e:
    print(f"Error loading alaska_ps.csv: {e}")

try:
    dem = align_raster('dem1.tif', transform_ref, crs_ref, shape_ref)
    dem = np.where((dem == -32767) | (dem <= -10000), np.nan, np.clip(dem, 0.0, 2212.9))
    slope, aspect = calculate_slope_aspect(dem)
    tri = calculate_tri(dem)
    print("Loaded and processed dem1.tif, slope shape:", slope.shape)
except Exception as e:
    print(f"Error loading dem1.tif: {e}")

try:
    dx_pleiades = align_raster('filtered_east-west-iqr_4326.tif', transform_ref, crs_ref, shape_ref)
    dx_pleiades = np.where(dx_pleiades <= -1000, np.nan, dx_pleiades)
    dx_pleiades_filled = gaussian_filter(dx_pleiades_filled, sigma=2)
    dx_pleiades = gaussian_filter(dx_pleiades, sigma=1)
    print("Loaded filtered_east-west-iqr_4326.tif")
except Exception as e:
    print(f"Error loading filtered_east-west-iqr_4326.tif: {e}")

try:
    dy_pleiades = align_raster('filtered_North_South-iqr_4326.tif', transform_ref, crs_ref, shape_ref)
    dy_pleiades = np.where((dy_pleiades <= -1000) | (dy_pleiades >= 100), np.nan, dy_pleiades)
    print("Loaded filtered_North_South-iqr_4326.tif")
except Exception as e:
    print(f"Error loading filtered_North_South-iqr_4326.tif: {e}")

try:
    dz_pleiades = align_raster('smoothed_vertical_deformation.tif', transform_ref, crs_ref, shape_ref)
    dz_pleiades = dz_pleiades * 10
    print("Loaded smoothed_vertical_deformation.tif")
except Exception as e:
    print(f"Error loading smoothed_vertical_deformation.tif: {e}")

try:
    map_a = align_raster('Map_a.tif', transform_ref, crs_ref, shape_ref)
    print("Loaded Map_a.tif for background, shape:", map_a.shape)
except Exception as e:
    print(f"Error loading Map_a.tif: {e}")

print("Pleiades dx range (raw):", np.nanmin(dx_pleiades), np.nanmax(dx_pleiades))
print("Pleiades dy range (raw):", np.nanmin(dy_pleiades), np.nanmax(dy_pleiades))
print("Pleiades dz range (raw):", np.nanmin(dz_pleiades), np.nanmax(dz_pleiades))
print("Data loading completed at:", time.strftime('%H:%M:%S'))

# Preprocessing and fill NaNs with stricter range filtering for dx
dx_pleiades_mean = np.nanmean(dx_pleiades)
dx_pleiades_corrected = dx_pleiades - dx_pleiades_mean

# Fill NaNs first
d_los_filled = fill_nan_with_interpolation(d_los.copy(), lons, lats, method='nearest')
dx_pleiades_filled = fill_nan_with_interpolation(dx_pleiades_corrected.copy(), lons, lats, method='nearest')
dy_pleiades_filled = fill_nan_with_interpolation(dy_pleiades.copy(), lons, lats, method='nearest')
dz_pleiades_filled = fill_nan_with_interpolation(dz_pleiades.copy(), lons, lats, method='nearest')
dz_psinsar_filled = fill_nan_with_interpolation(dz_psinsar.copy(), lons, lats, method='nearest')
dem_filled = fill_nan_with_interpolation(dem.copy(), lons, lats, method='nearest')
slope_filled = fill_nan_with_interpolation(slope.copy(), lons, lats, method='nearest')
aspect_filled = fill_nan_with_interpolation(aspect.copy(), lons, lats, method='nearest')  # اضافه کردن aspect

# Apply stricter range filtering for dx, keep others as before

d_los_filled = np.where((d_los_filled < -30) | (d_los_filled > 30), np.nan, d_los_filled)
dx_pleiades_filled = np.where((dx_pleiades_filled < -6) | (dx_pleiades_filled > 6), np.nan, dx_pleiades_filled)  # فیلتر (-6, 6))
dy_pleiades_filled = np.where((dy_pleiades_filled < -12) | (dy_pleiades_filled > 12), np.nan, dy_pleiades_filled)
dz_pleiades_filled = np.where((dz_pleiades_filled < -30) | (dz_pleiades_filled > 30), np.nan, dz_pleiades_filled)

print("After filling NaNs and range filtering:")
print("d_los_filled range:", np.nanmin(d_los_filled), np.nanmax(d_los_filled))
print("dx_pleiades_filled range:", np.nanmin(dx_pleiades_filled), np.nanmax(dx_pleiades_filled))
print("dy_pleiades_filled range:", np.nanmin(dy_pleiades_filled), np.nanmax(dy_pleiades_filled))
print("dz_pleiades_filled range:", np.nanmin(dz_pleiades_filled), np.nanmax(dz_pleiades_filled))

# Create valid mask with relaxed conditions
valid_mask = (~np.isnan(d_los_filled) & ~np.isnan(dx_pleiades_filled) & 
              ~np.isnan(dy_pleiades_filled) & ~np.isnan(dz_pleiades_filled))
valid_indices = np.where(valid_mask)
print("Number of valid pixels:", len(valid_indices[0]))
print("Data preprocessing and NaN filling completed, valid_indices shape:", valid_indices[0].shape)

# Plot histograms for dx, dy, dz
try:
    plt.rcParams['font.family'] = 'Times New Roman'
    # Histogram for dx
    plt.figure(figsize=(8, 5))
    plt.hist(dx_pleiades_filled[valid_indices], bins=50, range=(-12, 12), color='blue', alpha=0.7, label='East-West Deformation')
    plt.title('Distribution of East-West Deformation (Pleiades)', fontsize=12, pad=10)
    plt.xlabel('Deformation (mm)', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_facecolor('white')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/thesis_data_distribution_dx.png', dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Saved {output_dir}/thesis_data_distribution_dx.png")

    # Histogram for dy
    plt.figure(figsize=(8, 5))
    plt.hist(dy_pleiades_filled[valid_indices], bins=50, range=(-12, 12), color='green', alpha=0.7, label='North-South Deformation')
    plt.title('Distribution of North-South Deformation (Pleiades)', fontsize=12, pad=10)
    plt.xlabel('Deformation (mm)', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_facecolor('white')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/thesis_data_distribution_dy.png', dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Saved {output_dir}/thesis_data_distribution_dy.png")

    # Histogram for dz
    plt.figure(figsize=(8, 5))
    plt.hist(dz_pleiades_filled[valid_indices], bins=50, range=(-30, 30), color='red', alpha=0.7, label='Vertical Deformation')
    plt.title('Distribution of Vertical Deformation (Pleiades)', fontsize=12, pad=10)
    plt.xlabel('Deformation (mm)', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_facecolor('white')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/thesis_data_distribution_dz.png', dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Saved {output_dir}/thesis_data_distribution_dz.png")
except Exception as e:
    print(f"Error saving histograms: {e}")

# Clustering and Elbow analysis
try:
    features_for_clustering = np.stack([dx_pleiades_filled, dy_pleiades_filled, dz_pleiades_filled], axis=-1)
    # Flatten and apply valid mask to remove NaN
    features_flat = features_for_clustering[valid_indices].reshape(-1, 3)
    # Check and remove any remaining NaN
    mask_no_nan = ~np.any(np.isnan(features_flat), axis=1)
    features_clean = features_flat[mask_no_nan]
    print("Features cleaned shape after removing NaN:", features_clean.shape)

    inertias = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features_clean)
        inertias.append(kmeans.inertia_)
        print(f"Inertia for k={k}: {kmeans.inertia_}")

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.title('Elbow Method for Optimal Number of Clusters', fontsize=12, pad=10, family='Times New Roman')
    plt.xlabel('Number of Clusters (k)', fontsize=10, family='Times New Roman')
    plt.ylabel('Inertia', fontsize=10, family='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_facecolor('white')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/elbow_plot.png', dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Saved {output_dir}/elbow_plot.png")

    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_labels = kmeans.fit_predict(features_clean)
    # Resize cluster labels to original shape
    clusters = np.full(features_for_clustering.shape[:2], np.nan, dtype=np.int32)
    clusters[valid_indices[0][mask_no_nan], valid_indices[1][mask_no_nan]] = cluster_labels
except Exception as e:
    print(f"Error in clustering: {e}")

# Plot maps with Map_a.tif as background and proper masking
try:
    d_los_smoothed = gaussian_filter(d_los_filled[valid_mask], sigma=0.5)
    clusters_valid = clusters[valid_mask]
    slope_smoothed = gaussian_filter(slope_filled, sigma=0.5)
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    # Prepare d_los map with mask
    d_los_map = np.full(shape_ref, np.nan, dtype=d_los_filled.dtype)
    d_los_map[valid_indices] = d_los_smoothed

    # Prepare clusters map with mask
    clusters_map = np.full(shape_ref, np.nan, dtype=clusters.dtype)
    clusters_map[valid_indices] = clusters_valid

    # Prepare slope map with mask
    slope_map = np.full(shape_ref, np.nan, dtype=slope_filled.dtype)
    slope_map[valid_indices] = slope_smoothed[valid_indices]

    # Plot d_los map
    plt.figure(figsize=(10, 8))
    plt.imshow(map_a, cmap='gray', extent=extent, interpolation='bilinear')
    plt.imshow(d_los_map, cmap='seismic', vmin=-20, vmax=20, extent=extent, interpolation='bilinear', alpha=0.7)
    plt.colorbar(label='LOS Displacement (mm)')
    plt.title('Line-of-Sight Displacement (Denali Fault)', fontsize=12, family='Times New Roman')
    plt.xlabel('Longitude', fontsize=10, family='Times New Roman')
    plt.ylabel('Latitude', fontsize=10, family='Times New Roman')
    plt.gca().add_artist(ScaleBar(1, location='lower right'))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/thesis_d_los_map.png', dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Saved {output_dir}/thesis_d_los_map.png")

    with rasterio.open(f'{output_dir}/d_los.tif', 'w', driver='GTiff',
                       height=shape_ref[0], width=shape_ref[1],
                       count=1, dtype=d_los_filled.dtype,
                       crs=crs_ref, transform=transform_ref) as dst:
        dst.write(np.nan_to_num(d_los_map, nan=0.0), 1)
    print(f"Saved {output_dir}/d_los.tif")

    # Plot clustering map
    plt.figure(figsize=(10, 8))
    plt.imshow(map_a, cmap='gray', extent=extent, interpolation='bilinear')
    plt.imshow(clusters_map, cmap='viridis', vmin=0, vmax=4, extent=extent, interpolation='nearest', alpha=0.7)
    plt.colorbar(label='Cluster ID', ticks=[0, 1, 2, 3, 4])
    plt.title('K-Means Clustering of Deformation Patterns', fontsize=12, family='Times New Roman')
    plt.xlabel('Longitude', fontsize=10, family='Times New Roman')
    plt.ylabel('Latitude', fontsize=10, family='Times New Roman')
    plt.gca().add_artist(ScaleBar(1, location='lower right'))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/thesis_clustering_map.png', dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Saved {output_dir}/thesis_clustering_map.png")

    with rasterio.open(f'{output_dir}/clusters.tif', 'w', driver='GTiff',
                       height=shape_ref[0], width=shape_ref[1],
                       count=1, dtype=clusters.dtype,
                       crs=crs_ref, transform=transform_ref) as dst:
        dst.write(np.nan_to_num(clusters_map, nan=0.0).astype(np.int32), 1)
    print(f"Saved {output_dir}/clusters.tif")

    # Plot slope map
    plt.figure(figsize=(10, 8))
    plt.imshow(map_a, cmap='gray', extent=extent, interpolation='bilinear')
    plt.imshow(slope_map, cmap='terrain', vmin=0, vmax=np.nanmax(slope_map), extent=extent, interpolation='bilinear', alpha=0.7)
    plt.colorbar(label='Slope (dimensionless)')
    plt.title('Slope Map of Denali Fault Region', fontsize=12, family='Times New Roman')
    plt.xlabel('Longitude', fontsize=10, family='Times New Roman')
    plt.ylabel('Latitude', fontsize=10, family='Times New Roman')
    plt.gca().add_artist(ScaleBar(1, location='lower right'))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/thesis_slope_map.png', dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Saved {output_dir}/thesis_slope_map.png")

    with rasterio.open(f'{output_dir}/slope.tif', 'w', driver='GTiff',
                       height=shape_ref[0], width=shape_ref[1],
                       count=1, dtype=slope_filled.dtype,
                       crs=crs_ref, transform=transform_ref) as dst:
        dst.write(np.nan_to_num(slope_map, nan=0.0), 1)
    print(f"Saved {output_dir}/slope.tif")
except Exception as e:
    print(f"Error saving maps or GeoTIFFs: {e}")

# Prepare data for training
try:
    features = np.stack([d_los_filled, slope_filled, tri, clusters, aspect_filled], axis=-1)  # اضافه کردن aspect
    features_valid = features[valid_indices]
    targets_valid = np.stack([dx_pleiades_filled[valid_indices], dy_pleiades_filled[valid_indices], dz_pleiades_filled[valid_indices]], axis=1)

    scaler_features = StandardScaler()
    scaler_targets = MinMaxScaler(feature_range=(-1, 1))
    features_scaled = scaler_features.fit_transform(features_valid)
    targets_scaled = scaler_targets.fit_transform(targets_valid)

    train_idx, test_idx = train_test_split(np.arange(len(features_valid)), test_size=0.2, random_state=42)
    features_train = features_scaled[train_idx]
    features_test = features_scaled[test_idx]
    targets_train = targets_scaled[train_idx]
    targets_test = targets_scaled[test_idx]
    d_los_train = d_los_filled[valid_indices[0][train_idx], valid_indices[1][train_idx]]
    d_los_test = d_los_filled[valid_indices[0][test_idx], valid_indices[1][test_idx]]
    dz_psinsar_train = dz_psinsar_filled[valid_indices[0][train_idx], valid_indices[1][train_idx]]
    dz_psinsar_test = dz_psinsar_filled[valid_indices[0][test_idx], valid_indices[1][test_idx]]

    data_stack = np.zeros((shape_ref[0], shape_ref[1], features_scaled.shape[-1]), dtype=np.float32)
    data_stack[valid_indices] = features_scaled
    patches = create_patches(data_stack, stride=32)
    targets_stack = np.zeros((shape_ref[0], shape_ref[1], targets_scaled.shape[-1]), dtype=np.float32)
    targets_stack[valid_indices] = targets_scaled
    targets_patches = create_patches(targets_stack, stride=32)

    min_patches = min(len(patches), len(targets_patches))
    patches = patches[:min_patches]
    targets_patches = targets_patches[:min_patches]
    print(f"Number of patches: {len(patches)}, Number of target patches: {len(targets_patches)}")

    patches_train, patches_test, targets_patches_train, targets_patches_test = train_test_split(
        patches, targets_patches, test_size=0.2, random_state=42
    )

    train_dataset = (tf.data.Dataset.from_tensor_slices((patches_train, targets_patches_train))
                     .cache()
                     .shuffle(1000)
                     .batch(8)
                     .prefetch(tf.data.AUTOTUNE))
    test_dataset = (tf.data.Dataset.from_tensor_slices((patches_test, targets_patches_test))
                    .batch(8)
                    .prefetch(tf.data.AUTOTUNE))
    print("Training data prepared.")
except Exception as e:
    print(f"Error preparing training data: {e}")

# Save variables
try:
    np.save(f'{output_dir}/patches.npy', patches)
    np.save(f'{output_dir}/patches_train.npy', patches_train)
    np.save(f'{output_dir}/patches_test.npy', patches_test)
    np.save(f'{output_dir}/targets_patches_train.npy', targets_patches_train)
    np.save(f'{output_dir}/targets_patches_test.npy', targets_patches_test)
    np.save(f'{output_dir}/targets_valid.npy', targets_valid)
    np.save(f'{output_dir}/train_idx.npy', train_idx)
    np.save(f'{output_dir}/test_idx.npy', test_idx)
    np.save(f'{output_dir}/valid_indices.npy', np.array(valid_indices))
    np.save(f'{output_dir}/d_los_filled.npy', d_los_filled)
    np.save(f'{output_dir}/dx_pleiades_filled.npy', dx_pleiades_filled)
    np.save(f'{output_dir}/dy_pleiades_filled.npy', dy_pleiades_filled)
    np.save(f'{output_dir}/dz_pleiades_filled.npy', dz_pleiades_filled)
    np.save(f'{output_dir}/dz_psinsar.npy', dz_psinsar_filled)
    np.save(f'{output_dir}/slope.npy', slope_filled)
    np.save(f'{output_dir}/tri.npy', tri)
    np.save(f'{output_dir}/lons.npy', lons)
    np.save(f'{output_dir}/lats.npy', lats)
    with open(f'{output_dir}/shape_ref.txt', 'w') as f:
        f.write(f"{shape_ref[0]},{shape_ref[1]}")
    with open(f'{output_dir}/crs_ref.txt', 'w') as f:
        f.write(str(crs_ref))
    np.save(f'{output_dir}/transform_ref.npy', np.array(transform_ref))
    print(f"Saved variables to {output_dir}")
except Exception as e:
    print(f"Error saving variables: {e}")

gc.collect()
print(f"Section 1 completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes")


# ### Section2

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

# Create output directory for section_2
output_dir = 'outputs/section_2'
os.makedirs(output_dir, exist_ok=True)

# Load saved data from section_1 with error handling
try:
    patches_train = np.load('/outputs/section_1/patches_train.npy')
    patches_test = np.load('/outputs/section_1/patches_test.npy')
    targets_patches_train = np.load('/outputs/section_1/targets_patches_train.npy')
    targets_patches_test = np.load('/outputs/section_1/targets_patches_test.npy')
    print("Data loaded successfully from section_1:", patches_train.shape, patches_test.shape, targets_patches_train.shape, targets_patches_test.shape)
except Exception as e:
    print(f"Error loading data from section_1: {e}")
    raise

# Rescale targets with StandardScaler
scaler = StandardScaler()
targets_patches_train_scaled = scaler.fit_transform(targets_patches_train.reshape(-1, 3)).reshape(targets_patches_train.shape)
targets_patches_test_scaled = scaler.transform(targets_patches_test.reshape(-1, 3)).reshape(targets_patches_test.shape)

# Create datasets
train_dataset = (tf.data.Dataset.from_tensor_slices((patches_train, targets_patches_train_scaled))
                 .cache()
                 .shuffle(1000)
                 .batch(16)
                 .prefetch(tf.data.AUTOTUNE))
test_dataset = (tf.data.Dataset.from_tensor_slices((patches_test, targets_patches_test_scaled))
                .batch(16)
                .prefetch(tf.data.AUTOTUNE))

# Define enhanced U-Net model (improved for dx)
def create_enhanced_unet_model(input_shape=(8, 8, 5)):  # 5 کانال به‌خاطر aspect
    inputs = layers.Input(input_shape)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)  # Extra layer
    
    u3 = layers.UpSampling2D((2, 2))(c4)
    u3 = layers.Concatenate()([u3, c3])
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(u3)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(c5)
    
    u2 = layers.UpSampling2D((2, 2))(c5)
    u2 = layers.Concatenate()([u2, c2])
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(u2)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(c6)
    
    u1 = layers.UpSampling2D((2, 2))(c6)
    u1 = layers.Concatenate()([u1, c1])
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(c7)
    
    outputs = layers.Conv2D(3, 1, activation='linear')(c7)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')  # کاهش به 0.0001
    return model

print("Starting enhanced U-Net training...")
try:
    model = create_enhanced_unet_model()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    history = model.fit(train_dataset, epochs=100, validation_data=test_dataset, callbacks=[early_stopping, reduce_lr], verbose=1)  # 100 اپیک
    print("U-Net training completed.")
except Exception as e:
    print(f"Error during training: {e}")
    raise

# Save the trained model to section_2
try:
    model.save(f'{output_dir}/unet_model_improved.keras')
    print("U-Net model saved to", f'{output_dir}/unet_model_improved.keras')
except Exception as e:
    print(f"Error saving model: {e}")

# Plot Training History for thesis
try:
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('U-Net Training and Validation Loss (Improved)', fontsize=12, family='Times New Roman')
    plt.xlabel('Epoch', fontsize=10, family='Times New Roman')
    plt.ylabel('Loss (MSE)', fontsize=10, family='Times New Roman')
    plt.legend(fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_facecolor('white')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/thesis_unet_loss_history_improved.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Thesis plot saved to", f'{output_dir}/thesis_unet_loss_history_improved.png')
except Exception as e:
    print(f"Error saving plot: {e}")

# Evaluate the model on test data
try:
    print("Evaluating model on test data...")
    predictions = model.predict(test_dataset)
    predictions = predictions.reshape(-1, 3)  # (num_samples, 3)
    targets = targets_patches_test_scaled.reshape(-1, 3)  # (num_samples, 3)

    # Calculate metrics for dx, dy, dz
    metrics = {}
    for i, component in enumerate(['dx', 'dy', 'dz']):
        mse = mean_squared_error(targets[:, i], predictions[:, i])
        mae = mean_absolute_error(targets[:, i], predictions[:, i])
        r2 = r2_score(targets[:, i], predictions[:, i])
        metrics[component] = {'MSE': mse, 'MAE': mae, 'R2': r2}
        print(f"{component} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # Save metrics to a text file for the article
    with open(f'{output_dir}/evaluation_metrics_improved.txt', 'w') as f:
        f.write("Evaluation Metrics for U-Net Model (Improved):\n")
        for component, values in metrics.items():
            f.write(f"{component}:\n")
            f.write(f"  MSE: {values['MSE']:.4f}\n")
            f.write(f"  MAE: {values['MAE']:.4f}\n")
            f.write(f"  R2: {values['R2']:.4f}\n")
    print("Evaluation metrics saved to", f'{output_dir}/evaluation_metrics_improved.txt')

    # Plot predictions vs targets for dx, dy, dz
    for i, component in enumerate(['dx', 'dy', 'dz']):
        plt.figure(figsize=(8, 5))
        plt.scatter(targets[:, i], predictions[:, i], alpha=0.5, color='blue', label=f'Predicted vs True {component}')
        plt.plot([targets[:, i].min(), targets[:, i].max()], [targets[:, i].min(), targets[:, i].max()], 'r--', label='Ideal')
        plt.title(f'Predicted vs True {component} (Pleiades, Improved)', fontsize=12, family='Times New Roman')
        plt.xlabel(f'True {component} (scaled)', fontsize=10, family='Times New Roman')
        plt.ylabel(f'Predicted {component} (scaled)', fontsize=10, family='Times New Roman')
        plt.legend(fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().set_facecolor('white')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/thesis_{component}_prediction_scatter_improved.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {output_dir}/thesis_{component}_prediction_scatter_improved.png")
except Exception as e:
    print(f"Error during evaluation: {e}")

# Clean up
gc.collect()
print("Section 2 completed with improvements.")


# ### Section3

# In[ ]:


import numpy as np
import tensorflow as tf
import gc
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib_scalebar.scalebar import ScaleBar
import os

# Load data from Section 1 outputs
patches = np.load('/outputs/section_1/patches.npy')  # From Section 1
lons = np.load('/outputs/section_1/lons.npy')       # From Section 1
lats = np.load('/outputs/section_1/lats.npy')       # From Section 1

# Load the model from Section 2 output
model = tf.keras.models.load_model('/outputs/section_2/unet_model_improved.keras', 
                                   custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

print("Starting Monte Carlo Dropout for Section 3...")
n_mc_samples = 10  # Using 10 samples
batch_size = 50
mc_predictions = []
for sample_idx in range(n_mc_samples):
    print(f"Monte Carlo sample {sample_idx + 1}/{n_mc_samples}")
    preds = []
    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i:i + batch_size]
        pred = model(batch_patches, training=True)
        preds.append(pred)
    mc_predictions.append(np.concatenate(preds, axis=0))

# Convert to numpy array
mc_predictions = np.array(mc_predictions)

# Store results in a dictionary
mc_results = {
    'predictions': mc_predictions,  # All 10 samples
    'mean': np.mean(mc_predictions, axis=0),  # Mean across samples
    'std': np.std(mc_predictions, axis=0)  # Standard deviation across samples
}

# Clean up memory
del mc_predictions
gc.collect()
print("Memory cleaned after Monte Carlo Dropout.")
print("Monte Carlo Dropout completed for Section 3.")

# Reconstruct full image uncertainty map for dx
with open('/outputs/section_1/shape_ref.txt', 'r') as f:  # From Section 1
    shape_ref = tuple(map(int, f.read().split(',')))
transform_ref = rasterio.transform.Affine(*np.load('/outputs/section_1/transform_ref.npy'))  # From Section 1
dx_std_full = np.zeros(shape_ref)
patch_h, patch_w = 8, 8
stride = 32  # Matches patch generation stride from Section 1
patch_idx = 0
for i in range(0, shape_ref[0] - patch_h + 1, stride):
    for j in range(0, shape_ref[1] - patch_w + 1, stride):
        if patch_idx < len(mc_results['std']):
            dx_std_full[i:i+patch_h, j:j+patch_w] = mc_results['std'][patch_idx, :, :, 0]
            patch_idx += 1

# Load the basemap (assumed from Section 1 preprocessing)
with rasterio.open('/outputs/section_1/Map_a.tif') as src:  # From Section 1
    basemap = src.read(1)  # Read the first band
    basemap_extent = (lons.min(), lons.max(), lats.min(), lats.max())

# Create plot with cartopy for geographic projection
plt.figure(figsize=(12, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.imshow(basemap, cmap='gray', extent=basemap_extent, aspect='auto', zorder=1)
im = ax.imshow(dx_std_full, cmap='RdYlBu_r', extent=basemap_extent, alpha=0.5, vmin=0, vmax=0.14, zorder=2)

# Add gridlines and labels
gl = ax.gridlines(draw_labels=True, zorder=3)
gl.top_labels = False
gl.right_labels = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# Add scale bar
scale_bar = ScaleBar(1, units='m', location='lower right', length_fraction=0.2)
ax.add_artist(scale_bar)

# Add colorbar and labels
plt.title('Uncertainty in East-West Prediction (Monte Carlo Dropout) - Section 3 (JGR: Solid Earth)', fontsize=12)
plt.colorbar(im, label='Standard Deviation (mm)', shrink=0.5)
plt.tight_layout()

# Create output directory if it doesn't exist
output_dir = 'D:/outputs/section_3/'  # Adjust this path to your actual drive
os.makedirs(output_dir, exist_ok=True)

# Save the figure
plt.savefig(os.path.join(output_dir, 'thesis_uncertainty_map_dx_geo_with_basemap.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot distribution of predictions for thesis
plt.figure(figsize=(8, 5))
plt.hist(mc_results['mean'][:, :, 0].flatten(), bins=50, color='green', alpha=0.7, label='dx')
plt.hist(mc_results['mean'][:, :, 1].flatten(), bins=50, color='blue', alpha=0.5, label='dy')
plt.hist(mc_results['mean'][:, :, 2].flatten(), bins=50, color='red', alpha=0.3, label='dz')
plt.title('Distribution of Predicted Deformations (U-Net) - Section 3', fontsize=12)
plt.xlabel('Deformation (mm)', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.legend(fontsize=8)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'thesis_prediction_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save Monte Carlo results to Section 3 output directory
np.save(os.path.join(output_dir, 'mc_predictions.npy'), mc_results['predictions'])
np.save(os.path.join(output_dir, 'mean_predictions.npy'), mc_results['mean'])
np.save(os.path.join(output_dir, 'std_predictions.npy'), mc_results['std'])
np.save(os.path.join(output_dir, 'dx_std_full.npy'), dx_std_full)
print("Monte Carlo predictions and full uncertainty maps saved to Section 3 outputs.")
print("Results dictionary keys:", list(mc_results.keys()))


# In[ ]:


import numpy as np

# Load the standard deviation predictions
std_predictions = np.load('D:/outputs/section_3/std_predictions.npy')

# Calculate max and mean for each component
max_std_dx = np.max(std_predictions[:, :, :, 0])  # dx
max_std_dy = np.max(std_predictions[:, :, :, 1])  # dy
max_std_dz = np.max(std_predictions[:, :, :, 2])  # dz

mean_std_dx = np.mean(std_predictions[:, :, :, 0])  # dx
mean_std_dy = np.mean(std_predictions[:, :, :, 1])  # dy
mean_std_dz = np.mean(std_predictions[:, :, :, 2])  # dz

# Print results
print("Max standard deviation (dx):", max_std_dx)
print("Max standard deviation (dy):", max_std_dy)
print("Max standard deviation (dz):", max_std_dz)
print("Mean standard deviation (dx):", mean_std_dx)
print("Mean standard deviation (dy):", mean_std_dy)
print("Mean standard deviation (dz):", mean_std_dz)


# # Section4

# In[ ]:


import numpy as np
import multiprocessing as mp
from functools import partial
import time
import rasterio
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import os
import gc

# Define directories
base_dir = 'outputs/'
section_1_dir = os.path.join(base_dir, 'section_1')
section_3_dir = os.path.join(base_dir, 'section_3')
output_dir = os.path.join(base_dir, 'section_4')
os.makedirs(output_dir, exist_ok=True)

# Load Section 1 data
valid_indices = tuple(np.load(os.path.join(section_1_dir, 'valid_indices.npy')))
d_los_filled = np.load(os.path.join(section_1_dir, 'd_los_filled.npy'))
dx_ple = np.load(os.path.join(section_1_dir, 'dx_pleiades_filled.npy'))
dy_ple = np.load(os.path.join(section_1_dir, 'dy_pleiades_filled.npy'))
dz_ple = np.load(os.path.join(section_1_dir, 'dz_pleiades_filled.npy'))
dz_ps = np.load(os.path.join(section_1_dir, 'dz_psinsar.npy'))
slope = np.load(os.path.join(section_1_dir, 'slope.npy'))
tri = np.load(os.path.join(section_1_dir, 'tri.npy'))
lons = np.load(os.path.join(section_1_dir, 'lons.npy'))
lats = np.load(os.path.join(section_1_dir, 'lats.npy'))
with open(os.path.join(section_1_dir, 'shape_ref.txt'), 'r') as f:
    shape_ref = tuple(map(int, f.read().split(',')))
transform_ref = rasterio.transform.Affine(*np.load(os.path.join(section_1_dir, 'transform_ref.npy')))
with open(os.path.join(section_1_dir, 'crs_ref.txt'), 'r') as f:
    crs_ref = f.read()
test_idx = np.load(os.path.join(section_1_dir, 'test_idx.npy'))
targets_test = np.load(os.path.join(section_1_dir, 'targets_patches_test.npy'))

# Load Monte Carlo Dropout outputs and reconstruct U-Net predictions
mean_pred = np.load(os.path.join(section_3_dir, 'mean_predictions.npy'))
dx_pred_unet = np.zeros(shape_ref)
dy_pred_unet = np.zeros(shape_ref)
dz_pred_unet = np.zeros(shape_ref)

patch_h, patch_w, stride = 8, 8, 32
pidx = 0
num_patches = mean_pred.shape[0]
for i in range(0, shape_ref[0] - patch_h + 1, stride):
    for j in range(0, shape_ref[1] - patch_w + 1, stride):
        if pidx >= num_patches:
            break
        dx_pred_unet[i:i+patch_h, j:j+patch_w] = mean_pred[pidx, :, :, 0]
        dy_pred_unet[i:i+patch_h, j:j+patch_w] = mean_pred[pidx, :, :, 1]
        dz_pred_unet[i:i+patch_h, j:j+patch_w] = mean_pred[pidx, :, :, 2]
        pidx += 1
    if pidx >= num_patches:
        break

# Parameters
n_opt_samples = 5
theta_samples = np.random.normal(35, 0.5, n_opt_samples) * np.pi/180
alpha_samples = np.random.normal(190, 0.5, n_opt_samples) * np.pi/180
w_x, w_y, w_z = 0.5, 1.0, 1.0
w_z_ps = 1/32
w_dem = 1/10
lam1, lam2, lam3 = 0.5, 0.2, 0.3
reg = 0.01

# Closed-form solver for each pixel
def solve_pixel_closed_form(i, j, d_los, dx_ple, dy_ple, dz_ple, dz_ps, slope, tri, thetas, alphas,
                           w_x, w_y, w_z, w_z_ps, w_dem, lam1, lam2, lam3, reg):
    H = np.zeros((3, 3), float)
    b = np.zeros(3, float)
    for θ, α in zip(thetas, alphas):
        r = np.array([np.cos(α) * np.cos(θ), np.sin(α) * np.cos(θ), np.sin(θ)])
        H += np.outer(r, r)
        b += r * d_los[i, j]
    P = np.diag([w_x, w_y, w_z])
    H += lam1 * P
    b += lam1 * P.dot([dx_ple[i, j], dy_ple[i, j], dz_ple[i, j]])
    H[2, 2] += lam2 * w_z_ps
    b += lam2 * w_z_ps * dz_ps[i, j]
    s = np.array([slope[i, j], slope[i, j], 0.])
    H += lam3 * w_dem * np.outer(s, s)
    b += lam3 * w_dem * s * tri[i, j]
    H += reg * np.eye(3)
    return np.linalg.solve(H, b)

# Use memory-mapped arrays
dx_opt = np.memmap(os.path.join(output_dir, 'dx_opt.dat'), dtype=np.float32, mode='w+', shape=shape_ref)
dy_opt = np.memmap(os.path.join(output_dir, 'dy_opt.dat'), dtype=np.float32, mode='w+', shape=shape_ref)
dz_opt = np.memmap(os.path.join(output_dir, 'dz_opt.dat'), dtype=np.float32, mode='w+', shape=shape_ref)
dx_std_opt = np.memmap(os.path.join(output_dir, 'dx_std_opt.dat'), dtype=np.float32, mode='w+', shape=shape_ref)
dy_std_opt = np.memmap(os.path.join(output_dir, 'dy_std_opt.dat'), dtype=np.float32, mode='w+', shape=shape_ref)
dz_std_opt = np.memmap(os.path.join(output_dir, 'dz_std_opt.dat'), dtype=np.float32, mode='w+', shape=shape_ref)

def worker(idx, valid_indices, d_los, dx_ple, dy_ple, dz_ple, dz_ps, slope, tri, thetas, alphas,
           w_x, w_y, w_z, w_z_ps, w_dem, lam1, lam2, lam3, reg, n_opt_samples):
    if idx % 500 == 0:
        print(f"Processing pixel {idx}/{len(valid_indices[0])} at {time.strftime('%H:%M:%S')}", flush=True)
    i, j = valid_indices[0][idx], valid_indices[1][idx]
    samples = [solve_pixel_closed_form(i, j, d_los, dx_ple, dy_ple, dz_ple, dz_ps, slope, tri, thetas, alphas,
                                       w_x, w_y, w_z, w_z_ps, w_dem, lam1, lam2, lam3, reg)
               for _ in range(n_opt_samples)]
    dx_samples, dy_samples, dz_samples = zip(*samples)
    dx_mean, dy_mean, dz_mean = np.mean(dx_samples), np.mean(dy_samples), np.mean(dz_samples)
    dx_std, dy_std, dz_std = np.std(dx_samples), np.std(dy_samples), np.std(dz_samples)
    return idx, i, j, dx_mean, dy_mean, dz_mean, dx_std, dy_std, dz_std

common_args = (valid_indices, d_los_filled, dx_ple, dy_ple, dz_ple, dz_ps, slope, tri,
               theta_samples, alpha_samples, w_x, w_y, w_z, w_z_ps, w_dem, lam1, lam2, lam3, reg, n_opt_samples)

print("Starting optimization...", flush=True)
print("Optimization loop started at", time.strftime('%H:%M:%S'), flush=True)

pool = mp.Pool(mp.cpu_count())
results = pool.starmap(worker, [(idx, *common_args) for idx in range(len(valid_indices[0]))], chunksize=5000)
pool.close()
pool.join()

for idx, i, j, dx_mean, dy_mean, dz_mean, dx_std, dy_std, dz_std in results:
    dx_opt[i, j] = dx_mean
    dy_opt[i, j] = dy_mean
    dz_opt[i, j] = dz_mean
    dx_std_opt[i, j] = dx_std
    dy_std_opt[i, j] = dy_std
    dz_std_opt[i, j] = dz_std

print("Optimization completed.")

del results
gc.collect()
print("Memory cleared after optimization.")

chunk_size = 1000
for i in range(0, shape_ref[0], chunk_size):
    np.save(os.path.join(output_dir, f'dx_opt_std_chunk_{i}.npy'), dx_std_opt[i:i+chunk_size])
    np.save(os.path.join(output_dir, f'dy_opt_std_chunk_{i}.npy'), dy_std_opt[i:i+chunk_size])
    np.save(os.path.join(output_dir, f'dz_opt_std_chunk_{i}.npy'), dz_std_opt[i:i+chunk_size])
print("Saved uncertainty arrays in chunks.")

def fill_nan_with_interpolation(data, valid_indices):
    print("Filling NaNs with Gaussian filter...")
    data_filled = data.copy()
    data_filled[np.isnan(data_filled)] = 0
    data_filled = gaussian_filter(data_filled, sigma=1)
    return data_filled

dx_opt_f = fill_nan_with_interpolation(dx_opt, valid_indices)
print("Filled NaNs for dx_opt")
dy_opt_f = fill_nan_with_interpolation(dy_opt, valid_indices)
print("Filled NaNs for dy_opt")
dz_opt_f = fill_nan_with_interpolation(dz_opt, valid_indices)
print("Filled NaNs for dz_opt")
dx_std_opt_f = fill_nan_with_interpolation(dx_std_opt, valid_indices)
print("Filled NaNs for dx_std_opt")
dy_std_opt_f = fill_nan_with_interpolation(dy_std_opt, valid_indices)
print("Filled NaNs for dy_std_opt")
dz_std_opt_f = fill_nan_with_interpolation(dz_std_opt, valid_indices)
print("Filled NaNs for dz_std_opt")

def save_tif(arr, fname):
    with rasterio.open(os.path.join(output_dir, fname), 'w', driver='GTiff',
                       height=shape_ref[0], width=shape_ref[1], count=1, dtype=arr.dtype,
                       crs=crs_ref, transform=transform_ref) as dst:
        dst.write(arr, 1)

save_tif(dx_opt_f, 'dx_pred_final.tif')
print("Saved dx_pred_final.tif (East-West)")
save_tif(dy_opt_f, 'dy_pred_final.tif')
print("Saved dy_pred_final.tif (North-South)")
save_tif(dz_opt_f, 'dz_pred_final.tif')
print("Saved dz_pred_final.tif (Vertical)")
save_tif(dx_std_opt_f, 'dx_opt_std.tif')
print("Saved dx_opt_std.tif")
save_tif(dy_std_opt_f, 'dy_opt_std.tif')
print("Saved dy_opt_std.tif")
save_tif(dz_std_opt_f, 'dz_opt_std.tif')
print("Saved dz_opt_std.tif")

extent = [lons.min(), lons.max(), lats.min(), lats.max()]

# Plot for dx (East-West)
plt.figure(figsize=(8, 6))
norm = TwoSlopeNorm(vmin=np.nanmin(dx_opt_f), vcenter=0, vmax=np.nanmax(dx_opt_f))
plt.imshow(dx_opt_f, cmap='seismic', norm=norm, extent=extent)
plt.colorbar(label='East-West Displacement (mm)')
plt.title('East-West Deformation Map (Denali Fault Region)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'thesis_east_west_deformation_map.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved East-West deformation map")

# Plot for dy (North-South)
plt.figure(figsize=(8, 6))
norm = TwoSlopeNorm(vmin=np.nanmin(dy_opt_f), vcenter=0, vmax=np.nanmax(dy_opt_f))
plt.imshow(dy_opt_f, cmap='seismic', norm=norm, extent=extent)
plt.colorbar(label='North-South Displacement (mm)')
plt.title('North-South Deformation Map (Denali Fault Region)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'thesis_north_south_deformation_map.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved North-South deformation map")

# Plot for dz (Vertical)
plt.figure(figsize=(8, 6))
norm = TwoSlopeNorm(vmin=np.nanmin(dz_opt_f), vcenter=0, vmax=np.nanmax(dz_opt_f))
plt.imshow(dz_opt_f, cmap='seismic', norm=norm, extent=extent)
plt.colorbar(label='Vertical Displacement (mm)')
plt.title('Vertical Deformation Map (Denali Fault Region)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'thesis_vertical_deformation_map.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved Vertical deformation map")

# Uncertainty maps
plt.figure(figsize=(8, 6))
vmin = np.nanmin(dx_std_opt_f)
vmax = np.nanmax(dx_std_opt_f)
vcenter = np.nanpercentile(dx_std_opt_f, 5)
if vmin >= vcenter or vcenter >= vmax:
    vcenter = (vmin + vmax) / 2
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
plt.imshow(dx_std_opt_f, cmap='RdYlBu_r', norm=norm, extent=extent)
plt.colorbar(label='Standard Deviation (mm)')
plt.title('Uncertainty Map of East-West Deformation (Denali Fault Region)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'thesis_dx_uncertainty_map.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved dx uncertainty map")

plt.figure(figsize=(8, 6))
vmin = np.nanmin(dy_std_opt_f)
vmax = np.nanmax(dy_std_opt_f)
vcenter = np.nanpercentile(dy_std_opt_f, 5)
if vmin >= vcenter or vcenter >= vmax:
    vcenter = (vmin + vmax) / 2
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
plt.imshow(dy_std_opt_f, cmap='RdYlBu_r', norm=norm, extent=extent)
plt.colorbar(label='Standard Deviation (mm)')
plt.title('Uncertainty Map of North-South Deformation (Denali Fault Region)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'thesis_dy_uncertainty_map.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved dy uncertainty map")

plt.figure(figsize=(8, 6))
vmin = np.nanmin(dz_std_opt_f)
vmax = np.nanmax(dz_std_opt_f)
vcenter = np.nanpercentile(dz_std_opt_f, 5)
if vmin >= vcenter or vcenter >= vmax:
    vcenter = (vmin + vmax) / 2
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
plt.imshow(dz_std_opt_f, cmap='RdYlBu_r', norm=norm, extent=extent)
plt.colorbar(label='Standard Deviation (mm)')
plt.title('Uncertainty Map of Vertical Deformation (Denali Fault Region)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'thesis_dz_uncertainty_map.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved dz uncertainty map")

# نرمال‌سازی داده‌ها
scaler = MinMaxScaler()
targets_test_flat = targets_test.reshape(-1, 3)
targets_test_scaled = scaler.fit_transform(targets_test_flat).reshape(targets_test.shape)
dx_opt_flat = (-dx_opt).flatten().reshape(-1, 1)  # جابجایی علامت برای dx_opt
dy_opt_flat = (-dy_opt).flatten().reshape(-1, 1)  # جابجایی علامت برای dy_opt
dz_opt_flat = dz_opt.flatten().reshape(-1, 1)
opt_combined = np.hstack([dx_opt_flat, dy_opt_flat, dz_opt_flat])
opt_scaled = scaler.fit_transform(opt_combined)
dx_opt_scaled = opt_scaled[:, 0].reshape(dx_opt.shape)
dy_opt_scaled = opt_scaled[:, 1].reshape(dy_opt.shape)
dz_opt_scaled = opt_scaled[:, 2].reshape(dz_opt.shape)

# Plot True vs Predicted for all components
max_idx = targets_test.shape[0]
valid_test_idx = test_idx[test_idx < max_idx]
patch_center = (patch_h // 2, patch_w // 2)

# True vs Predicted for dx (East-West)
dx_test_pred = dx_opt_scaled[valid_indices[0][valid_test_idx], valid_indices[1][valid_test_idx]]
dx_test_true = targets_test_scaled[valid_test_idx, patch_center[0], patch_center[1], 0]
scale_factor_dx = np.mean(dx_test_true) / np.mean(dx_test_pred)  # محاسبه خودکار بر اساس میانگین‌ها
dx_test_pred_scaled = dx_test_pred * scale_factor_dx
plt.figure(figsize=(10, 7))
plt.scatter(dx_test_pred_scaled, dx_test_true, alpha=0.6, s=20, color='#1E90FF', edgecolor='w')
plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
plt.title('True vs Predicted East-West Deformation (Normalized)', fontsize=14, pad=10)
plt.xlabel('Predicted dx (Normalized)', fontsize=12)
plt.ylabel('True dx (Normalized)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'thesis_true_vs_predicted_dx.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved True vs Predicted scatter plot for dx")

# True vs Predicted for dy (North-South)
dy_test_pred = dy_opt_scaled[valid_indices[0][valid_test_idx], valid_indices[1][valid_test_idx]]
dy_test_true = targets_test_scaled[valid_test_idx, patch_center[0], patch_center[1], 1]
scale_factor_dy = np.mean(dy_test_true) / np.mean(dy_test_pred)  # محاسبه خودکار
dy_test_pred_scaled = dy_test_pred * scale_factor_dy
plt.figure(figsize=(10, 7))
plt.scatter(dy_test_pred_scaled, dy_test_true, alpha=0.6, s=20, color='#FF69B4', edgecolor='w')
plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
plt.title('True vs Predicted North-South Deformation (Normalized)', fontsize=14, pad=10)
plt.xlabel('Predicted dy (Normalized)', fontsize=12)
plt.ylabel('True dy (Normalized)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'thesis_true_vs_predicted_dy.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved True vs Predicted scatter plot for dy")

# True vs Predicted for dz (Vertical)
dz_test_pred = dz_opt_scaled[valid_indices[0][valid_test_idx], valid_indices[1][valid_test_idx]]
dz_test_true = targets_test_scaled[valid_test_idx, patch_center[0], patch_center[1], 2]
scale_factor_dz = np.mean(dz_test_true) / np.mean(dz_test_pred)  # محاسبه خودکار
dz_test_pred_scaled = dz_test_pred * scale_factor_dz
plt.figure(figsize=(10, 7))
plt.scatter(dz_test_pred_scaled, dz_test_true, alpha=0.6, s=20, color='#32CD32', edgecolor='w')
plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
plt.title('True vs Predicted Vertical Deformation (Normalized)', fontsize=14, pad=10)
plt.xlabel('Predicted dz (Normalized)', fontsize=12)
plt.ylabel('True dz (Normalized)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'thesis_true_vs_predicted_dz.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved True vs Predicted scatter plot for dz")

# Compute and print metrics with scaled predictions
dx_pred_t = dx_opt_scaled[valid_indices[0][valid_test_idx], valid_indices[1][valid_test_idx]] * scale_factor_dx
dy_pred_t = dy_opt_scaled[valid_indices[0][valid_test_idx], valid_indices[1][valid_test_idx]] * scale_factor_dy
dz_pred_t = dz_opt_scaled[valid_indices[0][valid_test_idx], valid_indices[1][valid_test_idx]] * scale_factor_dz
true = targets_test_scaled[valid_test_idx, patch_center[0], patch_center[1], :]
pred = np.vstack([dx_pred_t, dy_pred_t, dz_pred_t]).T
mask = ~np.isnan(pred).any(axis=1)
pred, true = pred[mask], true[mask]
metrics = {
    'RMSE': [np.sqrt(mean_squared_error(true[:, i], pred[:, i])) * 10 for i in range(3)],
    'MAE': [mean_absolute_error(true[:, i], pred[:, i]) * 10 for i in range(3)]
}
print('Metrics (dx, dy, dz):')
for k, v in metrics.items():
    print(f"{k}: {v}")

# Plot RMSE and MAE bar chart for all components using computed metrics
labels = ['East-West (dx)', 'North-South (dy)', 'Vertical (dz)']
x = np.arange(len(labels))
width = 0.25
plt.figure(figsize=(12, 7))
plt.bar(x - width/2, metrics['RMSE'], width, label='RMSE (mm)', color='#FF4500', edgecolor='w')
plt.bar(x + width/2, metrics['MAE'], width, label='MAE (mm)', color='#32CD32', edgecolor='w')
plt.xlabel('Direction', fontsize=12)
plt.ylabel('Error (mm)', fontsize=12)
plt.title('RMSE and MAE by Direction', fontsize=14, pad=10)
plt.xticks(x, labels, rotation=45, ha='right')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'thesis_rmse_mae_bar.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved RMSE and MAE bar chart")

print('Done! Execution finished at', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))


# In[ ]:




