from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import pandas as pd
import numpy as np
from Levenshtein import distance as levenshtein_distance

def find_closest_label(label, known_labels):
    # Convert label to string for consistent comparison
    label = str(label)
    distances = [levenshtein_distance(label, known_label) for known_label in known_labels]
    return known_labels[np.argmin(distances)]

def df_fit_transformer(df):
    oe_dict = {}
    df_copy = df.copy()

    # Encoding categorical variables
    for col in df_copy.select_dtypes(include=['object']).columns:
        oe = OrdinalEncoder()
        df_copy[[col]] = oe.fit_transform(df_copy[[col]].astype(str))
        oe_dict[col] = oe

    # Scaling numerical variables
    scaler = StandardScaler()
    df_copy[df_copy.select_dtypes(include=['number']).columns] = scaler.fit_transform(
        df_copy.select_dtypes(include=(['number'])))

    return df_copy, oe_dict, scaler

def df_transformer(df_to_transform, oe_dict, scaler):
    df_copy = df_to_transform.copy()

    # Function to transform columns
    def transform_column(col):
        oe = oe_dict[col.name]
        return col.map(lambda s: oe.transform([[str(s)]])[0][0] if str(s) in oe.categories_[0] else
        oe.transform([[find_closest_label(str(s), oe.categories_[0])]])[0][0])

    # Transforming categorical variables
    for col in df_copy.select_dtypes(include=['object']).columns:
        df_copy[col] = transform_column(df_copy[col].astype(str))

    # Scaling numerical variables
    numerical_columns = df_copy.select_dtypes(include=['number']).columns
    df_copy[numerical_columns] = scaler.transform(df_copy[numerical_columns])

    return df_copy
