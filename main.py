#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from sklearn.pipeline import Pipeline
from functions import check_missing_values, split_features_target, scale_features, select_k_best_anova, rfe_selection, sfs_selection, pca_selection, remove_correlated_features
from functions import plot_correlation_matrix

#%%
def main():
    # Load Data
    data = load_data()
    print(f'The number of samples: {len(data.index)}')
    print(f'The number of columns: {len(data.columns)}')

    # Check missing values
    check_missing_values(data)

    #Split dataset into features and encoded labels
    X, y = split_features_target(data)

    #Split into train and testset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    #Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("Train shape:", X_train_scaled.shape)
    print("Test shape:", X_test_scaled.shape)
    print("Label distribution training set:\n", y_train.value_counts())

    #Covariance feature elimination
    X_train_filtered, X_test_filtered, to_drop, surviving_cols = remove_correlated_features(X_train_scaled, X_test_scaled)

    #Plot the correlation matrix
    plot_correlation_matrix(X_train_scaled, to_drop, feature_names=X.columns.tolist())

    #%% Logistic Regression Classifier
    # Logistic Regression Classifier
    '''Needs feature selection beforehand, so we compare different methods to be able to use the best in the pipeline'''

    #%% SVM classifier
    #SVM classifier
    '''Needs feature selection beforehand, so we compare different methods to be able to use the best in the pipeline'''


    #%% XGBoost Classifier
    #XGBoost Classifier
    '''Doesn't need further feature selection, since this method handles that itself'''

    #%% Partial Least Square classifer
    #PLS classifier
    '''Doesn't need further feature selection, since this method handles that itself'''


    #%% Classifier Evaluation 

#%%
if __name__ == "__main__":
    main()


# %%
