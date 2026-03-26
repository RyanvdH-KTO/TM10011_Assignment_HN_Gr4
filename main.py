#%% Import packages
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from feature_engine.selection import DropCorrelatedFeatures
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from functions import check_missing_values, split_features_target
from functions import AUC_plot_and_confusion_matrix

from sklearn.preprocessing import FunctionTransformer


#%% Load Data
data = pd.read_csv('hn/Trainings_data.csv', index_col=0)

print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')
print(data['label'].value_counts())

#%%
def squeeze_output(X):
    # Ensure output is 2D
    if isinstance(X, tuple):
        X = X[0]
    return X.reshape(X.shape[0], -1)

#%% Main function
def main():

    scoring = "roc_auc"

    # Check missing values
    check_missing_values(data)

    # Split features/labels
    X, y = split_features_target(data)

    # Train/validation split
    X_train, X_validate, y_train, y_validate = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_validate.shape)
    print("Label distribution training set:\n", y_train.value_counts())

    # CV
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # --------------------------------------------------------------
    # PLS-DA Pipeline
    pipeline_pls_da = Pipeline([
        ('scaler', MinMaxScaler()),
        ('covariance_filter', DropCorrelatedFeatures(threshold=0.95)),
        ('pls', PLSRegression()),                 # dimensionality reduction
        ('squeeze', FunctionTransformer(squeeze_output)),
        ('classifier', LogisticRegression(       # converts to classification
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
    ])

    param_grid_pls_da = {
        'pls__n_components': [5, 10, 15],
        'classifier__C': [0.01, 0.1, 1, 10]
    }

    grid_search_pls_da = GridSearchCV(
        pipeline_pls_da,
        param_grid_pls_da,
        cv=kf,
        scoring=scoring,
        refit=True,
        n_jobs=-1
    )

    #Make extra CV for stability check
    '''Below we implement an outer CV, which is the grid search for the best hyperparameters,
    and the best classifier comes out. '''
    from sklearn.metrics import roc_curve, auc

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train_idx, val_idx in outer_cv.split(X_train, y_train):

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Create fresh GridSearch inside loop
        grid_search = GridSearchCV(
            pipeline_pls_da,
            param_grid_pls_da,
            cv=kf,
            scoring=scoring,
            refit=True,
            n_jobs=-1
        )

        grid_search.fit(X_tr, y_tr)

        best_model = grid_search.best_estimator_

        probs = best_model.predict_proba(X_val)[:, 1]

        fpr, tpr, _ = roc_curve(y_val, probs)

        # Interpolate for averaging
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0

        tprs.append(tpr_interp)
        aucs.append(auc(fpr, tpr))


    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # Fit model
    grid_search_pls_da.fit(X_train, y_train)

    classifier_PLS_DA = grid_search_pls_da.best_estimator_

    # Validation predictions
    y_pred_pls_da = classifier_PLS_DA.predict(X_validate)
    probabilities_pls_da = classifier_PLS_DA.predict_proba(X_validate)[:, 1]

    print('\nBest parameters found:\n', grid_search_pls_da.best_params_)
    print("Best CV score:", grid_search_pls_da.best_score_)
    print("\nClassification report (validation):\n",
          classification_report(y_validate, y_pred_pls_da, zero_division='warn'))

    AUC_plot_and_confusion_matrix(
        y_validate,
        probabilities_pls_da,
        y_validate,
        y_pred_pls_da,
        "PLS-DA model"
    )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,6))

    plt.plot(mean_fpr, mean_tpr,
            label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
            linewidth=2)

    plt.fill_between(mean_fpr,
                    mean_tpr - std_tpr,
                    mean_tpr + std_tpr,
                    alpha=0.2,
                    label="±1 std")

    plt.plot([0,1], [0,1], linestyle="--")

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve with STD (Outer CV)")
    plt.legend()
    plt.grid()

    plt.show()

    return classifier_PLS_DA


#%% Run training
if __name__ == "__main__":
    classifier_PLS_DA = main()


#%% Test set
test_data = pd.read_csv('hn/Test_data.csv', index_col=0)

print(f'\nTest samples: {len(test_data.index)}')
print(f'Test columns: {len(test_data.columns)}')
print(test_data['label'].value_counts())

# Preprocessing
check_missing_values(test_data)
X_test, y_test = split_features_target(test_data)

print("Test shape:", X_test.shape)
print("Label distribution test set:\n", y_test.value_counts())


#%% Test predictions
y_pred_pls_da = classifier_PLS_DA.predict(X_test)
probabilities_pls_da = classifier_PLS_DA.predict_proba(X_test)[:, 1]

print("\nClassification report (test):\n",
      classification_report(y_test, y_pred_pls_da, zero_division='warn'))

AUC_plot_and_confusion_matrix(
    y_test,
    probabilities_pls_da,
    y_test,
    y_pred_pls_da,
    "PLS-DA model",
    test=True
)