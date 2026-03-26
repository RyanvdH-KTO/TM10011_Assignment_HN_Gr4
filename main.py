#%% Import packages
# Import packages
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from functions import check_missing_values, split_features_target, scale_features, rfe_selection, sfs_selection, correlation_filter
from functions import AUC_plot_and_confusion_matrix
#%% Load Data
# Load Training Data
data = pd.read_csv('hn/Trainings_data.csv', index_col=0)
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')
print(data['label'].value_counts())

#%% Def Preprocessing & Classifier training
# Def Preprocessing & Classifier training
def main():
    # Determine scoring
    scoring = "accuracy"
    # Check missing values
    check_missing_values(data)

    #Split dataset into features and encoded labels
    X, y = split_features_target(data)

    #Split into train and validateset
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

    #Covariance feature elimination
#, to_drop, surviving_cols = remove_correlated_features(X_train_scaled, X_validate)

    # Define k-fold
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    #--------------------------------------------------------------
    # Pipeline Logistic regression
    pipeline_regression = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('covariance_filter', correlation_filter(threshold=0.95)),
        ('classifier', LogisticRegression(
                        penalty='l1',
                        solver='saga',
                        class_weight='balanced',
                        random_state=42,
                        max_iter=1000
                        ))
                        ])

    param_grid_regression = [{
        'classifier__C': [0.001, 0.01, 0.1, 1, 10],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear']
    },
    {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10],
        'classifier__penalty': ['elasticnet'],
        'classifier__solver': ['saga']
    }]
    print(X_validate.shape)
    grid_search_regression = GridSearchCV(pipeline_regression, param_grid_regression,
                                        cv=kf, scoring=scoring, refit = True, n_jobs=-1)
    '''
    # SFS Forward
    X_train_sfs_fwd_LR, X_validate_sfs_fwd_LR, indices_sfs_fwd_LR = sfs_selection(
        X_train_filtered,
        X_validate_filtered,
        y_train,
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        direction="forward",
        scoring=scoring,
        cv=kf
        )

    # SFS backward
    X_train_sfs_bwd_LR, X_validate_sfs_bwd_LR, indices_sfs_bwd_LR = sfs_selection(
        X_train_filtered,
        X_validate_filtered,
        y_train,
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        direction="backward",
        scoring = scoring,
        cv=kf
        )

    # RFE
    X_train_rfe_LR, X_validate_rfe_LR, indices_rfe_LR = rfe_selection(
        X_train_filtered,
        X_validate_filtered,
        y_train,
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        n_features=15
        )
    
    selector_data_LR = {
    "SFS_fwd": (X_train_sfs_fwd_LR, X_validate_sfs_fwd_LR, indices_sfs_fwd_LR),
    "SFS_bwd": (X_train_sfs_bwd_LR, X_validate_sfs_bwd_LR, indices_sfs_bwd_LR),
    "RFE": (X_train_rfe_LR, X_validate_rfe_LR, indices_rfe_LR)
    }

    best_selector_LR = max(selector_data_LR, key=lambda k: grid_search_regression.fit(selector_data_LR[k][0], y_train).score(selector_data_LR[k][1], y_validate))
    print(f"Best Selector: {best_selector_LR}")

    X_train_best_LR, X_validate_best_LR, LR_selector = selector_data_LR[best_selector_LR]
    '''
    grid_search_regression.fit(X_train, y_train)
    classifier_LR = grid_search_regression.best_estimator_
    print(X_validate.shape)
    y_pred_regression = classifier_LR.predict(X_validate)
    probabilities_regression = classifier_LR.predict_proba(X_validate)[:, 1]

    print('Best parameters found:\n', grid_search_regression.best_params_)
    print("Beste score:", grid_search_regression.best_score_)
    print(f"CL Report of LR:\n", classification_report(y_validate, y_pred_regression, zero_division='warn'))
    AUC_plot_and_confusion_matrix(y_validate, probabilities_regression, y_validate, y_pred_regression, "Logistic regression model")

    #--------------------------------------------------------------
    # Pipeline PLS-DA
    def squeeze_output(X):
        if isinstance(X, tuple):
            X = X[0]
        return X.reshape(X.shape[0], -1)

    pipeline_pls_da = Pipeline([
        ('scaler', MinMaxScaler()),
        ('pls', PLSRegression(n_components=1, scale=False, max_iter=10)),
        ('squeeze', FunctionTransformer(squeeze_output)),
        ('classifier', LogisticRegression(
                        penalty='elasticnet',
                        solver='saga',
                        class_weight='balanced',
                        l1_ratio=0.85,
                        random_state=42,
                        max_iter=1000
                        ))
                        ])

    param_grid_pls_da = {
        'pls__n_components': [5, 10, 15],
        'classifier__C': [0.001, 0.01, 0.1, 1, 10]
    }

    grid_search_pls_da = GridSearchCV(pipeline_pls_da, param_grid_pls_da, 
                                    cv=kf, scoring=scoring, refit = True, n_jobs=-1)
    
    grid_search_pls_da.fit(X_train_filtered, y_train)

    classifier_PLS_DA = grid_search_pls_da.best_estimator_ 
    y_pred_pls_da = classifier_PLS_DA.predict(X_validate_filtered)
    probabilities_pls_da = classifier_PLS_DA.predict_proba(X_validate_filtered)

    print('Best parameters found:\n', grid_search_pls_da.best_params_)
    print("Beste score:", grid_search_pls_da.best_score_)
    print(f"CL Report of PLS-DA:\n", classification_report(y_validate, y_pred_pls_da, zero_division='warn'))
    AUC_plot_and_confusion_matrix(y_validate, probabilities_pls_da[:,1], y_validate, y_pred_pls_da, "PLS DA model")

    #--------------------------------------------------------------
    # Pipeline Support Vector Machine
    pipeline_SVM = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('classifier', SVC(
                    random_state=42, 
                    max_iter=1000, 
                    class_weight='balanced', 
                    kernel = 'linear',
                    shrinking = True,
                    probability=True
                    ))               
                    ])
    
    param_grid_SVM = {
    'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'classifier__C': [0.0001, 0.001, 0.01, 1, 5, 10, 100, 1000],
    'classifier__gamma':['auto', 'scale', 0.0001, 0.001, 0.01, 1, 10, 100, 1000]
    }

    grid_search_SVM = GridSearchCV(pipeline_SVM, param_grid_SVM,
                                cv=kf, scoring=scoring, refit = True, n_jobs=-1)

    # SFS Forward
    X_train_sfs_fwd_SVM, X_validate_sfs_fwd_SVM, indices_sfs_fwd_SVM = sfs_selection(
        X_train_filtered,
        X_validate_filtered,
        y_train,
        estimator=SVC(max_iter=1000, random_state=42),
        direction="forward",
        scoring=scoring,
        cv=kf
        )

    # SFS backward
    X_train_sfs_bwd_SVM, X_validate_sfs_bwd_SVM, indices_sfs_bwd_SVM = sfs_selection(
        X_train_filtered,
        X_validate_filtered,
        y_train,
        estimator=SVC(max_iter=1000, random_state=42),
        direction="backward",
        scoring = scoring,
        cv=kf
        )

    # RFE
    X_train_rfe_SVM, X_validate_rfe_SVM, indices_rfe_SVM = rfe_selection(
        X_train_filtered,
        X_validate_filtered,
        y_train,
        estimator=SVC(kernel="linear", max_iter=1000, random_state=42),
        n_features=15
        )
    
    selector_data_SVM = {
    "SFS_fwd": (X_train_sfs_fwd_SVM, X_validate_sfs_fwd_SVM, indices_sfs_fwd_SVM),
    "SFS_bwd": (X_train_sfs_bwd_SVM, X_validate_sfs_bwd_SVM, indices_sfs_bwd_SVM),
    "RFE": (X_train_rfe_SVM, X_validate_rfe_SVM, indices_rfe_SVM)
    }

    best_selector_SVM = max(selector_data_SVM, key=lambda k: grid_search_SVM.fit(selector_data_SVM[k][0], y_train).score(selector_data_SVM[k][1], y_validate))
    print(f"Best Selector: {best_selector_SVM}")

    X_train_best_SVM, X_validate_best_SVM, SVM_selector = selector_data_SVM[best_selector_SVM]
    grid_search_SVM.fit(X_train_best_SVM, y_train)

    classifier_SVM = grid_search_SVM.best_estimator_ 
    y_pred_SVM = classifier_SVM.predict(X_validate_best_SVM) 
    probabilities_SVM = classifier_SVM.predict_proba(X_validate_best_SVM)

    print('Best parameters found:\n', grid_search_SVM.best_params_)
    print("Beste score:", grid_search_SVM.best_score_)
    print(f"CL Report of SVM:\n", classification_report(y_validate, y_pred_SVM, zero_division='warn'))
    AUC_plot_and_confusion_matrix(y_validate, probabilities_SVM[:,1], y_validate, y_pred_SVM, "Support vector machine")
    
    #--------------------------------------------------------------
    # Pipeline Gradient Boosting
    pipeline_XGB = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('classifier', xgb.XGBClassifier(
                        random_state=42,
                        n_estimators=1000,
                        max_depth=10,
                        learning_rate=0.1
                        )) 
                        ])
    
    param_grid_XGB = {
    'classifier__n_estimators': [100, 300, 500],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0]}

    grid_search_XGB = GridSearchCV(pipeline_XGB, param_grid_XGB, 
                                cv=kf, scoring=scoring, refit = True, n_jobs=-1)

    grid_search_XGB.fit(X_train_filtered, y_train)

    classifier_XGB = grid_search_XGB.best_estimator_ 
    y_pred_XGB = classifier_XGB.predict(X_validate_filtered)  
    propabilities_XGB = classifier_XGB.predict_proba(X_validate_filtered)[:, 1]
    if propabilities_XGB.ndim == 1:
        propabilities_XGB = np.column_stack([1 - propabilities_XGB, propabilities_XGB])

    print('Best parameters found:\n', grid_search_XGB.best_params_)
    print("Beste score:", grid_search_XGB.best_score_)
    print(f"CL Report of XGB:\n", classification_report(y_validate, y_pred_XGB, zero_division='warn'))
    AUC_plot_and_confusion_matrix(y_validate, propabilities_XGB[:,1], y_validate, y_pred_XGB, "XGBoost model")

    return X_train, classifier_LR, classifier_PLS_DA, classifier_SVM, classifier_XGB, LR_selector, SVM_selector

#%% Run model
# Run model
if __name__ == "__main__":
    X_train, classifier_LR, classifier_PLS_DA, classifier_SVM, classifier_XGB, LR_selector, SVM_selector = main()

#%% Test with Test Data
# Load Test Data
test_data = pd.read_csv('hn/Test_data.csv', index_col=0)
print(f'The number of samples: {len(test_data.index)}')
print(f'The number of columns: {len(test_data.columns)}')
print(test_data['label'].value_counts())

# Preprocessing
check_missing_values(test_data)
X_test, y_test = split_features_target(test_data)
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
X_train_filtered, X_test_filtered, to_drop, surviving_cols = remove_correlated_features(X_train_scaled, X_test_scaled)
X_test_selected_LR = X_test_filtered[:, LR_selector]
X_test_selected_SVM = X_test_filtered[:, SVM_selector]

print("Test shape:", X_test_filtered.shape)
print("Test LR shape", X_test_selected_LR.shape)
print("Test SVM shape", X_test_selected_SVM.shape)
print("Label distribution training set:\n", y_test.value_counts())

#%%
# LR test
y_pred_regression = classifier_LR.predict(X_test_selected_LR)
probabilities_regression = classifier_LR.predict_proba(X_test_selected_LR)[:, 1]
print(y_pred_regression.shape)
print(f"CL Report of LR:\n", classification_report(y_test, y_pred_regression, zero_division='warn'))
AUC_plot_and_confusion_matrix(y_test, probabilities_regression, y_test, y_pred_regression, "Logistic regression model", test=True)

# PLS-DA test
y_pred_pls_da = classifier_PLS_DA.predict(X_test_filtered)
probabilities_pls_da = classifier_PLS_DA.predict_proba(X_test_filtered)

print(f"CL Report of PLS-DA:\n", classification_report(y_test, y_pred_pls_da, zero_division='warn'))
AUC_plot_and_confusion_matrix(y_test, probabilities_pls_da[:,1], y_test, y_pred_pls_da, "PLS DA model", test=True)

# SVM test
y_pred_SVM = classifier_SVM.predict(X_test_selected_SVM) 
probabilities_SVM = classifier_SVM.predict_proba(X_test_selected_SVM)

print(f"CL Report of SVM:\n", classification_report(y_test, y_pred_SVM, zero_division='warn'))
AUC_plot_and_confusion_matrix(y_test, probabilities_SVM[:,1], y_test, y_pred_SVM, "Support vector machine", test=True)
    
# XGB test
y_pred_XGB = classifier_XGB.predict(X_test_filtered)  
propabilities_XGB = classifier_XGB.predict_proba(X_test_filtered)[:, 1]
if propabilities_XGB.ndim == 1:
    propabilities_XGB = np.column_stack([1 - propabilities_XGB, propabilities_XGB])

print(f"CL Report of XGB:\n", classification_report(y_test, y_pred_XGB, zero_division='warn'))
AUC_plot_and_confusion_matrix(y_test, propabilities_XGB[:,1], y_test, y_pred_XGB, "XGBoost model", test=True)


# %%