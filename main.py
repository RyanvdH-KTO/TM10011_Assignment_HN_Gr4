#%% Import packages
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from feature_engine.selection import DropCorrelatedFeatures
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from functions import check_missing_values, split_features_target, AUC_plot_and_confusion_matrix, ROC_STD_plot

#%% Function to squeeze output from PLS -- LG doesnt work > 2D
def squeeze_output(X):
    if isinstance(X, tuple):
        X = X[0]
    return X.reshape(X.shape[0], -1)

#%% Main function
def main():    #%% Load Data
    data = pd.read_csv('hn/Trainings_data.csv', index_col=0)
    print(f'The number of samples: {len(data.index)}')
    print(f'The number of columns: {len(data.columns)}')
    print(data['label'].value_counts())

    scoring = "roc_auc"

    # Check missing values
    check_missing_values(data)

    # Split features/labels
    X, y = split_features_target(data)

    # Train/validation split
    X_train, X_validate, y_train, y_validate = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_validate.shape)
    print("Label distribution training set:\n", y_train.value_counts())

    # Inner CV for GridSearch
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Outer CV for ROC ± std
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # --------------------------------------------------------------
    # PLS-DA Pipeline
    pipeline_pls_da = Pipeline([
        ('scaler', MinMaxScaler()),
        ('covariance_filter', DropCorrelatedFeatures(threshold=0.95)),
        ('pls', PLSRegression()),
        ('squeeze', FunctionTransformer(squeeze_output)),
        ('classifier', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
    ])

    param_grid_pls_da = {
        'pls__n_components': [5, 10, 15],
        'classifier__C': [0.01, 0.1, 1, 10]
    }

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train_idx, val_idx in outer_cv.split(X_train, y_train):
        # Proper indexing for DataFrame/Series
        if isinstance(X_train, pd.DataFrame):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        else:
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Fresh GridSearch for inner CV
        grid_search = GridSearchCV(
            pipeline_pls_da,
            param_grid_pls_da,
            cv=inner_cv,
            scoring=scoring,
            refit=True,
            n_jobs=-1
        )

        grid_search.fit(X_tr, y_tr)
        best_model = grid_search.best_estimator_
        probs = best_model.predict_proba(X_val)[:, 1]

        fpr, tpr, _ = roc_curve(y_val, probs)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0

        tprs.append(tpr_interp)
        aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # Fit final model on full training set
    final_grid = GridSearchCV(
        pipeline_pls_da,
        param_grid_pls_da,
        cv=inner_cv,
        scoring=scoring,
        refit=True,
        n_jobs=-1
    )
    final_grid.fit(X_train, y_train)
    classifier_PLS_DA = final_grid.best_estimator_

    # Validation predictions
    y_pred_pls_da = classifier_PLS_DA.predict(X_validate)
    probabilities_pls_da = classifier_PLS_DA.predict_proba(X_validate)[:, 1]

    print('\nBest parameters found:\n', final_grid.best_params_)
    print("\nClassification report (validation):\n",
          classification_report(y_validate, y_pred_pls_da, zero_division='warn'))

    AUC_plot_and_confusion_matrix(
        y_validate, probabilities_pls_da, y_validate, y_pred_pls_da, "PLS-DA model"
    )

    # Plot ROC ± std
    ROC_STD_plot(mean_fpr, mean_tpr, mean_auc, std_auc, std_tpr)

    # --------------------------------------------------------------
    # Pipeline Logistic regression
    pipeline_regression = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('covariance_filter', DropCorrelatedFeatures(threshold=0.95)),
        ('selector', SFS),
        ('classifier', LogisticRegression(
                        penalty='l1',
                        solver='saga',
                        class_weight='balanced',
                        random_state=42,
                        max_iter=10000
                        ))
                        ])

    param_grid_regression = [{
        'selector': [RFE(LogisticRegression(max_iter=1000, random_state=42), n_features_to_select=15), 
                    SFS(LogisticRegression(max_iter=1000, random_state=42), n_features_to_select="auto", direction="forward", scoring=scoring, n_jobs=-1, tol=1e-3),
                    SFS(LogisticRegression(max_iter=1000, random_state=42), n_features_to_select="auto", direction="backward", scoring=scoring, n_jobs=-1, tol=1e-3)],
        'classifier__C': [0.001, 0.01, 0.1, 1, 10],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear']
    },
    {
        'selector': [RFE(LogisticRegression(max_iter=1000, random_state=42), n_features_to_select=15), 
                    SFS(LogisticRegression(max_iter=1000, random_state=42), n_features_to_select="auto", direction="forward", scoring=scoring, n_jobs=-1, tol=1e-3),
                    SFS(LogisticRegression(max_iter=1000, random_state=42), n_features_to_select="auto", direction="backward", scoring=scoring, n_jobs=-1, tol=1e-3)], 
        'classifier__C': [0.001, 0.01, 0.1, 1, 10],
        'classifier__penalty': ['elasticnet'],
        'classifier__solver': ['saga']
    }]

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train_idx, val_idx in outer_cv.split(X_train,y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        grid_search_regression = GridSearchCV(pipeline_regression, param_grid_regression,
                                        cv=inner_cv, scoring=scoring, refit = True, n_jobs=-1)
        grid_search_regression.fit(X_tr, y_tr)
        best_model = grid_search_regression.best_estimator_

        probs = best_model.predict_proba(X_val)[:, 1]

        fpr, tpr, _ = roc_curve(y_val, probs)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0

        tprs.append(tpr_interp)
        aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    #Fit final model
    final_grid = GridSearchCV(pipeline_regression, param_grid_regression,
                                        cv=inner_cv, scoring=scoring, refit = True, n_jobs=-1)
    final_grid.fit(X_train,y_train)
    classifier_LR = final_grid.best_estimator_

    #Validation predictions
    y_pred_regression = classifier_LR.predict(X_validate)
    probabilities_regression = classifier_LR.predict_proba(X_validate)[:, 1]
    
    print('Best parameters found:\n', grid_search_regression.best_params_)
    print(f"CL Report of LR:\n", classification_report(y_validate, y_pred_regression, zero_division='warn'))
    
    AUC_plot_and_confusion_matrix(
        y_validate, probabilities_regression, y_validate, y_pred_regression, "Logistic regression model"
        )
    
    ROC_STD_plot(mean_fpr, mean_tpr, mean_auc, std_auc, std_tpr)

    return classifier_PLS_DA, classifier_LR

#%% Run training
if __name__ == "__main__":
    classifier_PLS_DA, classifier_LR = main()

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
    y_test, probabilities_pls_da, y_test, y_pred_pls_da, "PLS-DA model", test=True
)
# %%
