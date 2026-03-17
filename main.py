#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from hn.load_data import load_data
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_validate



#%% Load Data
# Load Data
# data = load_data()
# print(f'The number of samples: {len(data.index)}')
# print(f'The number of columns: {len(data.columns)}')

# data = pd.DataFrame(data)

#%%
#Load data

def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'HN_radiomicFeatures.csv'), index_col=0)

    return data


# %%
# Missing data functie
def check_missing_values(data):

    missing = data.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing data")
    else:
        print("Missing values per column:")
        print(missing)

    return missing


# %%
# Data splitten in features (X) en target (y)
def split_features_target(data, label_col='label'):
    X = data.drop(columns=[label_col]) 
    y = data[label_col]
    return X, y


# %%
# Encode lables: T12 = 0, T34 = 1
def encode_labels(y):
    y_encoded = y.map({'T12': 0, 'T34': 1})
    return y_encoded


#%%
# Train test split (stratified)
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test

#%%
# Feature scaling
# Meerdere scalars vergelijken: kijken welke beste accuracy geeft (en dan bv standard als baseline gebruiken)
def scale_features(X_train, X_test, method="standard"):

    if method == "standard":
        scaler = StandardScaler()

    elif method == "minmax":
        scaler = MinMaxScaler()

    elif method == "robust":
        scaler = RobustScaler()

    else:
        raise ValueError("Unknown scaler")

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler



# %%

def preprocess_pipeline(scale_method="standard"):

    # Load dataset
    data = load_data()

    # Check missing values
    check_missing_values(data)

    # Split features and target
    X, y = split_features_target(data)

    print("Feature shape:", X.shape)
    print("Original label distribution:")
    print(y.value_counts())

    # Encode labels
    y = encode_labels(y)

    print("Encoded label distribution:")
    print(y.value_counts())

    # Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Train shape before scaling:", X_train.shape)
    print("Test shape before scaling:", X_test.shape)

    # Feature scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, method=scale_method
    )

    print("Final shape training:", X_train_scaled.shape)
    print("Final shape test:", X_test_scaled.shape)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


#%%
from feature_extraction import select_k_best_anova, sfs_selection
def main():
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(scale_method="standard")
    return X_train, X_test, y_train, y_test, scaler


#%%
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = main()

# %%
