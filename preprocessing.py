# src/preprocessing.py
from data_columns import categorical_columns, numerical_columns

from sklearn.preprocessing import LabelEncoder
import shutil
import os
import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def change_data_types(dataframe):
    """Change data type of categorical features to string."""
    for col in categorical_columns:
        dataframe[col] = dataframe[col].astype('string')
    return dataframe

def clean_units(dataframe):
    """Remove units from specific columns and convert them to float."""
    columns_with_units = ['Nutrient N Sensor (ppm)', 'Nutrient P Sensor (ppm)', 'Nutrient K Sensor (ppm)']
    for column in columns_with_units:
        dataframe[column] = dataframe[column].str.replace(' ppm', '').astype(float)
    return dataframe

def remove_duplicates(dataframe):
    """Remove duplicate rows from the DataFrame."""
    duplicate_rows = dataframe[dataframe.duplicated()]
    print(f"Number of duplicate rows: {duplicate_rows.shape[0]}")
    if not duplicate_rows.empty:
        print("Duplicate rows:")
        print(duplicate_rows)
    dataframe = dataframe.drop_duplicates()
    return dataframe

def standardize_columns(dataframe):
    """Standardize values in specific columns."""
    dataframe['Plant Type'] = dataframe['Plant Type'].replace({
        'vine crops': 'Vine Crops',
        'VINE CROPS': 'Vine Crops',
        'herbs': 'Herbs',
        'HERBS': 'Herbs',
        'leafy greens': 'Leafy Greens',
        'LEAFY GREENS': 'Leafy Greens',
        'fruiting vegetables': 'Fruiting Vegetables',
        'FRUITING VEGETABLES': 'Fruiting Vegetables'
    })

    dataframe['Plant Stage'] = dataframe['Plant Stage'].replace({
        'maturity': 'Maturity',
        'MATURITY': 'Maturity',
        'seedling': 'Seedling',
        'SEEDLING': 'Seedling',
        'vegetative': 'Vegetative',
        'VEGETATIVE': 'Vegetative'
    })
    return dataframe

def combine_columns(dataframe):
    """Combine Plant Type and Plant Stage into a new column."""
    dataframe['Plant Type-Stage'] = dataframe['Plant Type'] + '-' + dataframe['Plant Stage']
    return dataframe

def handle_negatives_and_missing(dataframe):
    """Replace negative values and impute missing data with medians."""
    columns_to_fix = ['Light Intensity Sensor (lux)', 'Temperature Sensor (°C)']
    for column in columns_to_fix:
        # Replace negatives with medians
        dataframe[column] = dataframe.groupby('Plant Type-Stage')[column].transform(
            lambda x: x.mask(x < 0, x[x >= 0].median())
        )
        # Impute missing values with medians
        dataframe[column] = dataframe.groupby('Plant Type-Stage')[column].transform(
            lambda x: x.fillna(x.median())
        )
    return dataframe

def label_encode_columns(dataframe):
    """Encode categorical columns with LabelEncoder."""
    label_encoders = {col: LabelEncoder() for col in categorical_columns}
    for col in categorical_columns:
        dataframe[col] = label_encoders[col].fit_transform(dataframe[col])
    return dataframe, label_encoders

def preprocess_data(dataframe):
    """Master function to preprocess the data."""
    dataframe = change_data_types(dataframe)
    dataframe = clean_units(dataframe)
    dataframe = remove_duplicates(dataframe)
    dataframe = standardize_columns(dataframe)
    dataframe = combine_columns(dataframe)
    dataframe = handle_negatives_and_missing(dataframe)
    dataframe, label_encoders = label_encode_columns(dataframe)
    return dataframe
