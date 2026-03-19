import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import PCA



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

# Split data into features (X) and target (y)
def split_features_target(data, label_col='label'):
    X = data.drop(columns=[label_col]) 
    y = data[label_col]
    #eEncode labels: T12 = 0, T34 = 1
    y = y.map({'T12': 0, 'T34': 1})
    return X, y

# Feature scaling
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

def remove_correlated_features(X_train, X_test, threshold=0.95):
    # calculate correlation matrix on train only!
    corr_matrix = pd.DataFrame(X_train).corr().abs()

    # find upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # find columns where correlation exceeds threshold
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    print(f"Removing {len(to_drop)} correlated features")
    print(f"Features remaining: {X_train.shape[1] - len(to_drop)}")

    # drop from both train and test
    X_train_filtered = pd.DataFrame(X_train).drop(columns=to_drop).values
    X_test_filtered = pd.DataFrame(X_test).drop(columns=to_drop).values

    return X_train_filtered, X_test_filtered, to_drop

def plot_correlation_matrix(X_train, feature_names=None):
    corr_matrix = pd.DataFrame(X_train, columns=feature_names).corr().abs()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        center=0,
        vmin=0, vmax=1,
        square=True,
        linewidths=0,
        cbar_kws={"shrink": 0.8, "label": "Absolute Correlation"},
    )
    plt.title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()

#%% Feature selectors

# Univariate feature selection: kiest de beste k features op basis van hun individuele relatie met de target
def select_k_best_anova(X_train, X_test, y_train, k=10): #k zegt hoeveel features je wil
    selector = SelectKBest(score_func=f_classif, k=k) #SelectKBest: kijkt naar alle features afzonderlijk, geeft elke feature een score, en kiest daarna de k hoogste scores
                                                      # f_classif: betekent dat je de ANOVA F-test gebruikt
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    # selected_indices = selector.get_support(indices=True) #welke features gekozen zijn
    # scores = selector.scores_ #ANOVA scores per feature

    # print("Selected feature indices:", selected_indices)
    # print("Scores of all features:", scores)
    # print("Shape train after selection:", X_train_sel.shape)
    # print("Shape test after selection:", X_test_sel.shape)

    return X_train_sel, X_test_sel

# %%
# RFE: recursive feature elimination
def rfe_selection(X_train, X_test, y_train, estimator, n_features=10):

    selector = RFE(estimator=estimator, n_features_to_select=n_features)

    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    #selected_indices = selector.get_support(indices=True)
    #ranking = selector.ranking_

    #print("Selected feature indices:", selected_indices)
    #print("Ranking of all features:", ranking) #Ranking zegt iets over in welke ronde de feature eruit is gegooid
    #print("Shape train after selection:", X_train_sel.shape)
    #print("Shape test after selection:", X_test_sel.shape)

    return X_train_sel, X_test_sel

# Sequential Feature Selector
def sfs_selection(X_train, X_test, y_train, estimator, n_features=10, direction="forward", scoring="accuracy", cv=5):
    selector = SequentialFeatureSelector(
        estimator,
        n_features_to_select=n_features,
        direction=direction,
        scoring=scoring,
        cv=cv,
        n_jobs=-1
    )

    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    # selected_indices = selector.get_support(indices=True)

    # print("Selected feature indices:", selected_indices)
    # print("Shape train after selection:", X_train_sel.shape)
    # print("Shape test after selection:", X_test_sel.shape)

    return X_train_sel, X_test_sel

#PCA feature selection 
def pca_selection(X_train, X_test, n_components=0.95):
    pca = PCA(n_components=n_components)
    
    X_train_pca = pca.fit_transform(X_train)  # fit on train only
    X_test_pca = pca.transform(X_test)        # apply to test

    return X_train_pca, X_test_pca

