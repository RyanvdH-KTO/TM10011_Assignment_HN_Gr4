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

def ROC_STD_plot(mean_fpr, mean_tpr, mean_auc, std_auc, std_tpr):
    plt.figure(figsize=(8,6))
    plt.plot(mean_fpr, mean_tpr,
             label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
             linewidth=2)
    plt.fill_between(mean_fpr,
                     mean_tpr - std_tpr,
                     mean_tpr + std_tpr,
                     alpha=0.2, label="±1 std")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("FPR"), plt.ylabel("TPR"), plt.title("ROC Curve with STD")
    plt.legend(), plt.grid(), plt.show()
# %%