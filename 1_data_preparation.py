# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:58:37 2024

@author: mageo
"""

import os
import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt



def process_time_series(input_file_path, output_file_path, prefix, data_type):
    
    df_tall = pd.read_csv(input_file_path)

    # Check for necessary columns based on data_type
    if data_type == 'crop':
        if 'crpname_eg' not in df_tall.columns:
            raise KeyError("The 'crpname_eg' column is missing from the input DataFrame.")
        index_columns = ['id', 'crpname_eg', 'lat', 'lon']
        

    df_tall['id'] = range(1, len(df_tall) + 1)

    # Get the list of dates
    all_dates = df_tall['date'].unique().tolist()

    # Create a pivot table to reshape the data
    pivot_df = df_tall.pivot(index=index_columns, columns='date', values=prefix)

    # Reset index to make 'id' and 'crop_name' columns
    pivot_df.reset_index(inplace=True)

    # Reindex to ensure 'id', 'crop_name', and dates are in the correct order
    pivot_df = pivot_df.reindex(columns=index_columns + all_dates)

    # Move up the values in each column to eliminate NaNs
    df_moved_up = pivot_df.apply(lambda x: x.dropna().reset_index(drop=True))

    # Replace original columns with moved-up values
    for col in pivot_df.columns[2:]:
        pivot_df[col] = df_moved_up[col]

    # Drop rows with NaN values
    pivot_df = pivot_df.dropna()

    # Save the transformed DataFrame to a new CSV file
    pivot_df.to_csv(output_file_path, index=False)

    return pivot_df


def add_prefix_to_features(directory, output_path, drop_cols=None):
    files = glob.glob(f"{directory}/*_wide.csv")

    # Initialize an empty dictionary to hold DataFrames
    dataframes = {}

    # Loop through each file and read it into a DataFrame
    for file_path in files:
        if 'VV' in file_path:
            dtype = 'VV'
        elif 'VH' in file_path:
            dtype = 'VH'
        elif 'NDVI' in file_path:
            dtype = 'NDVI'
        else:
            # Skip files that don't match expected types
            continue
        
        df = pd.read_csv(file_path)
        # Add prefix to columns that start with a number
        df.columns = [f'{dtype}_{col}' if col[0].isdigit() else col for col in df.columns]
        dataframes[dtype] = df

    # Ensure all required DataFrames are present
    if not all(key in dataframes for key in ['VV', 'VH', 'NDVI']):
        raise ValueError("All three datasets (VV, VH, NDVI) are required.")

    # Merge DataFrames on shared columns
    index_crop = ['id', 'crpname_eg', 'lat', 'lon']
    gt_crop = pd.merge(
        pd.merge(dataframes['VV'], dataframes['VH'], on=index_crop, how='inner'),
        dataframes['NDVI'], on=index_crop, how='inner'
    )

    # Remove rows containing -9999.000000
    gt_crop = gt_crop[~(gt_crop.isin([-9999.000000]).any(axis=1))]

    # Drop specified columns, if any
    if drop_cols:
        gt_crop.drop(drop_cols, axis=1, inplace=True)

    # Save the processed DataFrame to a CSV file
    gt_crop.to_csv(output_path, index=False)
    return gt_crop


def assign_class_values(input_file, output_file_path):
    #gt_crop = pd.read_csv(input_file_path)
    gt_crop= input_file
    
    crop_map = {'Wheat': 1, 'Gram': 2, 'Mustard': 3}
    gt_crop['Class'] = gt_crop['crpname_eg'].map(crop_map).fillna(0).astype(int)
    
    # Drop unnecessary columns
    cols_to_drop = ['crpname_eg', 'lat', 'lon']
    gt_crop = gt_crop.drop(cols_to_drop, axis=1)
    
    # Reset index
    gt_crops3inc = gt_crop.drop('id', axis=1)
    gt_crops3inc.reset_index(drop=True)

    # Save the final DataFrame to a new CSV file
    gt_crops3inc.to_csv(output_file_path, index=False)
    return gt_crops3inc




def plot_timeseries(df, prefix, file_name):
    
    crop_names = {1: "Wheat", 2: "Gram", 3: "Mustard"}
    crops_to_plot = [1, 2, 3]
    crop_dfs = {}

    for crop in crops_to_plot:
        df_crop = df[df['Class'] == crop]
        df_crop = df_crop.filter(like=prefix)
        df_cleaned = df_crop[~(df_crop.isin([-9999.000000]).any(axis=1))]

        mean_values = df_cleaned.mean(axis=0)
        std_values = df_cleaned.std(axis=0)
        crop_dfs[crop] = (mean_values, std_values)

    plt.figure(figsize=(12, 10), dpi=80)
    colors = ['Red', 'Green', 'Blue']

    # Loop through crops and plot
    for idx, crop in enumerate(crops_to_plot):
        mean_values, std_values = crop_dfs[crop]
        plt.plot(mean_values, marker=None, color=colors[idx], label=f'{crop_names[crop]} Mean')
        plt.fill_between(
            range(len(mean_values)),
            mean_values - std_values,
            mean_values + std_values,
            color=colors[idx], alpha=0.2, label=f'{crop_names[crop]} Standard Deviation'
        )

    # Customize plot
    plt.legend(loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(prefix.strip('_'))
    plt.xlabel('Date')
    plt.title(f"{prefix.strip('_')} Timeseries for Crops Wheat, Gram, Mustard", fontweight='bold')
    plt.grid()

    # Save the plot as a PNG
    plt.savefig(file_name)
    plt.close()
    print(f"Plot saved as {file_name}")





# Main data preparation flow
dir_path = r'D:\Crop\Ujjain\Data\timeseries\*.csv'
crop_timeseries = glob.glob(dir_path)

# Step 1: Process time series data for each file
for i in crop_timeseries:
    input_file_path = i
    
    # Extract the base file path and extension from the file path
    base, ext = os.path.splitext(input_file_path)
    output_file_path = f"{base}_wide{ext}"
    
    
    # Extract the base name of the file path (last component of the path)
    base_name = os.path.basename(input_file_path)
    prefix = base_name.split('_')[0]
    print(f"Running conversion for {i} with {prefix}")
    
    process_time_series(input_file_path, output_file_path, prefix, data_type='crop')


# Step 2: Merge VV, VH, and NDVI datasets
input_dir = r'D:\Crop\Ujjain\Data\timeseries'
output_file = r'D:\Crop\Ujjain\Data\timeseries\timeseries_crops3inc.csv'
drop_cols=['NDVI_2023-10-31']
gt_crop_3bands = add_prefix_to_features(directory=input_dir, output_path=output_file, drop_cols=drop_cols)

# Step 3: Assign class values (crop labels)
#input_file_path = r'D:\Crop\Ujjain\Data\timeseries\timeseries_crops3inc.csv'
output_file_path = r'D:\Crop\Ujjain\Data\timeseries\crops3inc_multiclass_S1S2_onlytimeseries_v01.csv'
final_csv = assign_class_values(input_file=gt_crop_3bands, output_file_path=output_file_path)

# Step 4: Plot timeseries data
#df = pd.read_csv(r"D:\Crop\Ujjain\Data\timeseries\crops3inc_multiclass_S1S2_onlytimeseries_v01.csv")
prefixes = ['VV_', 'VH_', 'NDVI_']

for prefix in prefixes:
    plot_timeseries(final_csv, prefix, fr'D:\Crop\Ujjain\Data\timeseries_trend_figures\{prefix}Timeseries.png')
