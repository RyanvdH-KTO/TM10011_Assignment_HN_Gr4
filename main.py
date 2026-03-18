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
from feature_extraction import select_k_best_anova, rfe_selection, sfs_selection 
from sklearn.pipeline import Pipeline

#%% Function definitions
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


# Data splitten in features (X) en target (y)
def split_features_target(data, label_col='label'):
    X = data.drop(columns=[label_col]) 
    y = data[label_col]
    #encode labels: T12 = 0, T34 = 1
    y = y.map({'T12': 0, 'T34': 1})
    return X, y

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

    #Pipeline that compares feature selectors and classifiers
    feature_selectors = {
        "k best ANOVA" : SelectKBest(score_func=f_classif, k=10),
        "RFE"          : RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=5),
    }

    classifiers = {
        "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    }

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for sel_name, selector in feature_selectors.items():
        for clf_name, clf in classifiers.items():
            
            selector.fit(X_train_scaled, y_train)
            selected_indices = selector.get_support(indices=True)
            selected_names = X.columns[selected_indices].tolist()
            print(f"\n[{sel_name}] Selected features: {selected_names}")

            pipe = Pipeline([
                ("selector", selector),
                ("clf",      clf),
            ])
            
            #Get different cross-validation scores
            scores = cross_validate(
                pipe, X_train_scaled, y_train,
                cv=cv,
                scoring=["accuracy", "roc_auc", "f1"],
            )

            results.append({
                "Feature Selection" : sel_name,
                "Classifier"        : clf_name,
                "Accuracy"          : scores["test_accuracy"].mean(),
                "ROC-AUC"           : scores["test_roc_auc"].mean(),
                "F1"                : scores["test_f1"].mean(),
            })

    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df, X_test, y_test


    

#%%
if __name__ == "__main__":
    results_df, X_test, y_test = main()


# %%
