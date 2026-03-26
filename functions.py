# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix 
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

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
    #Encode labels: T12 = 0, T34 = 1
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

def remove_highly_correlated_features(X, threshold=0.95):
    """Remove features with correlation higher than the threshold."""
    df = pd.DataFrame(X)
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    
    return df.drop(columns=to_drop).values

def make_correlation_filter(threshold=0.95, validate=False):
    # This will return a transformer function for FunctionTransformer
    cols_to_keep = []

    def fit_transform(X, y=None):
        nonlocal cols_to_keep
        df = pd.DataFrame(X)
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        cols_to_keep = [col for col in df.columns if col not in to_drop]
        return df[cols_to_keep].values

    def transform(X):
        df = pd.DataFrame(X)
        return df[cols_to_keep].values

    return FunctionTransformer(func=transform, validate=False, kw_args={})

def plot_correlation_matrix(X_train, to_drop, feature_names=None):
    df = pd.DataFrame(X_train, columns=feature_names)
    
    # only keep the columns that will be dropped
    df_dropped = df[to_drop]
    corr_matrix = df_dropped.corr().abs()

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
    plt.title("Correlation Matrix — Dropped Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()

# Feature selectors

# RFE: recursive feature elimination
def rfe_selection(X_train, X_test, y_train, estimator, n_features=10):

    selector = RFE(estimator=estimator, n_features_to_select=n_features)

    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    selected_indices = selector.get_support(indices=True)

    return X_train_sel, X_test_sel, selected_indices

# Sequential Feature Selector
def sfs_selection(X_train, X_test, y_train, estimator, direction="forward", scoring="accuracy", cv=5): # accuracy kan ook met f1 of ROC-AUC, dat nog goed beargumenteren
    selector = SequentialFeatureSelector(
        estimator=estimator,
        n_features_to_select="auto",
        direction=direction,
        scoring=scoring,
        tol=1e-3,
        cv=cv,
        n_jobs=-1
    )

    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    selected_indices = selector.get_support(indices=True)

    return X_train_sel, X_test_sel, selected_indices


# Plot AUC-curve & confusion matrix

def AUC_plot_and_confusion_matrix(labels, probs, y_test, y_pred, model, test=False):
    # info
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(labels.values.ravel(), probs.ravel())
    roc_auc = auc(fpr, tpr)
    
    if test == True:
        model = model + " with testset"

    fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(fpr, tpr, color='blue', linewidth=2.5, label=f'AUC: {roc_auc:.3f}', linestyle='solid')
    ax1.plot([0, 1], [0, 1], color='grey', linestyle=(0, (5, 10)), label='Random prediction')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax1.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax1.set_title(f'ROC Curve\n{model}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid()
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', ax=ax2,
                xticklabels=['T12','T34'], 
                yticklabels=['T12','T34'],
                cbar_kws={'label': 'Count'})
    ax2.set_title(f'Confusion Matrix\n{model}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('Actual', fontsize=12)
    
    plt.tight_layout()
    plt.show()

    return roc_auc
# %%