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
    # plot_correlation_matrix(X_train_scaled, to_drop, feature_names=X.columns.tolist())

    # Feature selection
    # Deze uitvoeren voor estimator = logistic regresion en voor de SVM. 

    # Logistic Regression estimator for feature selection
    lr_estimator = LogisticRegression(
        penalty='l2',
        solver='liblinear',
        class_weight='balanced',
        random_state=42,
        max_iter=10000
    )
    # Deze gebruikte ik even om te testen, maar hier moet dan goeie LR model of SVM

    # Feature selection
    # Select k best ANOVA
    X_train_anova, X_test_anova = select_k_best_anova(
    X_train_filtered,
    X_test_filtered,
    y_train,
    k=10
    )

    # print("Shape train after k best ANOVA:", X_train_anova.shape)
    # print("Shape test after ANOVA:", X_test_anova.shape)


    # SFS forward
    X_train_sfs_fwd, X_test_sfs_fwd = sfs_selection(
    X_train_filtered,
    X_test_filtered,
    y_train,
    estimator=lr_estimator,
    direction="forward",
    scoring="accuracy", #kan ook roc-auc
    cv=5
    ) 

    # print("Shape train after SFS forward:", X_train_sfs_fwd.shape)
    # print("Shape test after SFS forward:", X_test_sfs_fwd.shape)

    # SFS backward
    X_train_sfs_bwd, X_test_sfs_bwd = sfs_selection(
        X_train_filtered,
        X_test_filtered,
        y_train,
        estimator=lr_estimator,
        direction="backward",
        scoring="accuracy",
        cv=5
    )  

    # print("Shape train after SFS backward:", X_train_sfs_bwd.shape)
    # print("Shape test after SFS backward:", X_test_sfs_bwd.shape)

    # RFE
    X_train_rfe, X_test_rfe = rfe_selection(
        X_train_filtered,
        X_test_filtered,
        y_train,
        estimator=lr_estimator,
        n_features=10
    )

    # print("Shape train after RFE:", X_train_rfe.shape)
    # print("Shape test after RFE:", X_test_rfe.shape)

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
