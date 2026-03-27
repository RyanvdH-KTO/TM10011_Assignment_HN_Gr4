# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 

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

def summarize(arr):
    arr = np.array(arr)
    return arr.mean(), np.percentile(arr, 2.5), np.percentile(arr, 97.5)         

def Bootstrap_calculation(y_test, probabilities, y_pred):
    n_bootstrap = 5000
    rng = np.random.default_rng(42)

    tprs = []
    aucs = []
    accs = []
    precs = []
    recs = []
    f1s = []
    mean_fpr = np.linspace(0, 1, 100)

    y_true = y_test.values

    for _ in range(n_bootstrap):
        # sample with replacement
        indices = rng.integers(0, len(y_true), len(y_true))
        
        y_sample = y_true[indices]
        prob_sample = probabilities[indices]
        pred_sample = y_pred[indices]

        if len(np.unique(y_sample)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_sample, prob_sample)
        
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0

        tprs.append(tpr_interp)
        aucs.append(auc(fpr, tpr))
        accs.append(accuracy_score(y_sample, pred_sample))
        precs.append(precision_score(y_sample, pred_sample, zero_division=0))
        recs.append(recall_score(y_sample, pred_sample, zero_division=0))
        f1s.append(f1_score(y_sample, pred_sample, zero_division=0))

    # Aggregate
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # 95% CI for AUC
    ci_lower = np.percentile(aucs, 2.5)
    ci_upper = np.percentile(aucs, 97.5)

    acc_mean, acc_lo, acc_hi = summarize(accs)
    prec_mean, prec_lo, prec_hi = summarize(precs)
    rec_mean, rec_lo, rec_hi = summarize(recs)
    f1_mean, f1_lo, f1_hi = summarize(f1s)

    print(f"AUC: {mean_auc:.3f} (95% CI: {ci_lower:.3f} – {ci_upper:.3f})")
    print("\nBootstrap Classification Metrics (95% CI):\n")
    print(f"Accuracy  : {acc_mean:.3f} (95% CI: {acc_lo:.3f} – {acc_hi:.3f})")
    print(f"Precision : {prec_mean:.3f} (95% CI: {prec_lo:.3f} – {prec_hi:.3f})")
    print(f"Recall    : {rec_mean:.3f} (95% CI: {rec_lo:.3f} – {rec_hi:.3f})")
    print(f"F1-score  : {f1_mean:.3f} (95% CI: {f1_lo:.3f} – {f1_hi:.3f})")
    
    return mean_fpr, mean_tpr, mean_auc, std_auc, std_tpr

def ROC_STD_plot(mean_fpr, mean_tpr, mean_auc, std_auc, std_tpr, model):
    plt.figure(figsize=(8,6))
    plt.plot(mean_fpr, mean_tpr,
             label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
             linewidth=2)
    plt.fill_between(mean_fpr,
                     mean_tpr - std_tpr,
                     mean_tpr + std_tpr,
                     alpha=0.2, label="±1 std")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("FPR"), plt.ylabel("TPR"), plt.title(f"ROC Curve with STD\n{model}")
    plt.legend(), plt.grid(), plt.show()
# %%