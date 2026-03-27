#%% Import packages
# Import packages
import numpy as np                                                                         # inport package for nummeric calculations and array operations 
import pandas as pd                                                                        # import package reading and handling tabular data
import xgboost as xgb                                                                      # import package for XGBoost classification
from sklearn.svm import SVC                                                                # import package fpr support vector machine classification
from sklearn.pipeline import Pipeline                                                      # import package for building preporcessing and modeling pipelines
from sklearn.feature_selection import RFE                                                  # import package for recursive feature elimination
from sklearn.preprocessing import MinMaxScaler                                             # import package scaling features to a fixed range
from sklearn.metrics import classification_report                                          # import package for evaluating classification models
from sklearn.linear_model import LogisticRegression                                        # import package for logistic regression classsification
from sklearn.cross_decomposition import PLSRegression                                      # import package for partial least squares modeling
from feature_engine.selection import DropCorrelatedFeatures                                # import package for removing highly correlated features
from sklearn.feature_selection import SequentialFeatureSelector as SFS                     # import package for sequential feature selection
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV        # import package for data splitting, cross-validation and grid search
from functions import check_missing_values, split_features_target                          # import zelf gebouwde functie
from functions import AUC_plot_and_confusion_matrix                                        # import zelf gebouwde functie

#%% Load Data
data = pd.read_csv('hn/Trainings_data.csv', index_col=0)                                   # read the csv with 
print(f'The number of samples: {len(data.index)}')                                         # print the number of samples. The number of rows equal the number of samples
print(f'The number of features: {len(data.columns)}')                                      # print the number of features/ The number of columns correspond to the number of features
print(data['label'].value_counts())                                                        # print datatype

#%% Def Preprocessing & Classifier training
def main():                                                                                # define the Definition function
    scoring = "roc_auc"                                                                    # define a variable to use for scoring. Use ROC-AUC for scoring
    check_missing_values(data)                                                             # check for missing values 
    X, y = split_features_target(data)                                                     # split dataset into features and encoded labels
    X_train, X_validate, y_train, y_validate = train_test_split(                           # split into train and validateset
        X,                                                                                 # feature matrix
        y,                                                                                 # target vector
        test_size=0.2,                                                                     # 20% of data to validationset 
        stratify=y,                                                                        # keep classification in both sets
        random_state=42)                                                                   # same seed for reproducibility

    print("Train shape:", X_train.shape)                                                   # print the shape of the training set
    print("Validation shape:", X_validate.shape)                                           # print the shape of the validation set
    print("Label distribution training set:\n", y_train.value_counts())                    # print the label distribution of the training set

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)                       # define k-fold

    # Pipeline Logistic regression -------------------------------------------------------
    pipeline_regression = Pipeline(steps=[                                                 # define the pipeline
        ('scaler', MinMaxScaler()),                                                        # scale features by minmaxscaler method
        ('covariance_filter', DropCorrelatedFeatures(threshold=0.95)),                     # filter for covariance, drop features which are correlated higher than 0.95
        ('selector', SFS),                                                                 # placeholder for featureselector in grid search
        ('classifier', LogisticRegression(                                                 # define classifier
                        penalty='l1',                                                      # penalty settings
                        solver='saga',                                                     # solver that support elasticnet
                        class_weight='balanced',                                           # correct for skewed class distribution
                        random_state=42,                                                   # same seed for reproducibility
                        max_iter=10000))])                                                 # define max iterations
                        
    param_grid_regression = [{                                                             # define the grid-parameter
        'selector': [RFE(LogisticRegression(                                               # RFE as  estimator option
                                            max_iter=1000,                                 # define max iterations
                                            random_state=42),                              # same seed for reproducibility
                                            n_features_to_select=15),                      # select this amount of the most important features to keep
                    SFS(LogisticRegression(                                                # SFS forward as estimator option 
                                            max_iter=1000,                                 # define max iterations
                                            random_state=42),                              # same seed for reproducibility
                                            n_features_to_select="auto",                   # automatically determine the optimal number of features to keep
                                            direction="forward",                           # add the feature one by one till performance doesn't improve
                                            scoring=scoring,                               # scoring is based on the earlier defines variable: ROC-AUC
                                            cv=kf,                                         # same as earlier defines variable: stratified K fold cross-validation
                                            n_jobs=-1,                                     # use all avaiable CPU cores
                                            tol=1e-3),                                     # stop if improvement is smaller than this value
                    SFS(LogisticRegression(                                                # 
                                            max_iter=1000,                                 # define max iterations
                                            random_state=42),                              # same seed for reproducibility
                                            n_features_to_select="auto",                   # automatically determine the optimal number of features to keep
                                            direction="backward",                          # remove a feature one by one till performance worsen beyond a threshold
                                            scoring=scoring,                               # scoring is based on the earlier defines variable: ROC-AUC
                                            cv=kf,                                         # same as earlier defines variable: stratified K fold cross-validation
                                            n_jobs=-1,                                     # use all avaiable CPU cores
                                            tol=1e-3)],                                    # stop if improvement is smaller than this value
        'classifier__C': [0.001, 0.01, 0.1, 1, 10],                                        #
        'classifier__penalty': ['l1', 'l2'],                                               #
        'classifier__solver': ['liblinear']                                                #
    }, {                                                                                   # choose between these options, with different in penaly and solvers
        'selector': [RFE(LogisticRegression(                                               # RFE as  estimator option
                                            max_iter=1000,                                 # define max iterations
                                            random_state=42),                              # same seed for reproducibility
                                            n_features_to_select=15),                      # select this amount of the most important features to keep
                    SFS(LogisticRegression(                                                # SFS forward as estimator option
                                            max_iter=1000,                                 # define max iterations
                                            random_state=42),                              # same seed for reproducibility
                                            n_features_to_select="auto",                   # automatically determine the optimal number of features to keep
                                            direction="forward",                           # add the feature one by one till performance doesn't improve
                                            scoring=scoring,                               # scoring is based on the earlier defines variable: ROC-AUC
                                            cv=kf,                                         # same as earlier defines variable: stratified K fold cross-validation
                                            n_jobs=-1,                                     # use all avaiable CPU cores
                                            tol=1e-3),                                     # stop if improvement is smaller than this value
                    SFS(LogisticRegression(                                                # SFS backward as estimator option
                                            max_iter=1000,                                 # define max iterations
                                            random_state=42),                              # same seed for reproducibility
                                            n_features_to_select="auto",                   # automatically determine the optimal number of features to keep
                                            direction="backward",                          # remove a feature one by one till performance worsen beyond a threshold
                                            scoring=scoring,                               # scoring is based on the earlier defines variable: ROC-AUC
                                            cv=kf,                                         # same as earlier defines variable: stratified K fold cross-validation
                                            n_jobs=-1,                                     # use all avaiable CPU cores
                                            tol=1e-3)],                                    # stop if improvement is smaller than this value
        'classifier__C': [0.001, 0.01, 0.1, 1, 10],                                        #
        'classifier__penalty': ['elasticnet'],                                             #
        'classifier__solver': ['saga']}]                                                   #
   
    grid_search_regression = GridSearchCV(                                                 #
        pipeline_regression,                                                               #
        param_grid_regression,                                                             #
        cv=kf,                                                                             # same as earlier defines variable: stratified K fold cross-validation
        scoring=scoring,                                                                   # scoring is based on the earlier defines variable: ROC-AUC
        refit = True, 
        n_jobs=-1)                                                                         # use all avaiable CPU cores

    grid_search_regression.fit(X_train, y_train)                                           #
    classifier_LR = grid_search_regression.best_estimator_                                 #

    y_pred_regression = classifier_LR.predict(X_validate)
    probabilities_regression = classifier_LR.predict_proba(X_validate)[:, 1]
    
    print('Best parameters found:\n', grid_search_regression.best_params_)
    print("Beste score:", grid_search_regression.best_score_)
    print(f"CL Report of LR:\n", classification_report(
        y_validate, y_pred_regression, zero_division='warn'))
    AUC_plot_and_confusion_matrix(y_validate, probabilities_regression, 
                                  y_validate, y_pred_regression, 
                                  "Logistic regression model")

    # Pipeline PLS-DA--------------------------------------------------------------
    pipeline_pls_da = Pipeline([
        ('scaler', MinMaxScaler()),
        ('covariance_filter', DropCorrelatedFeatures(threshold=0.95)),
        ('classifier', PLSRegression(
                                     n_components=10, 
                                     random_state=42))])                                   # same seed for reproducibility

    param_grid_pls_da = {
        'classifier__n_components': [5, 10, 15]}

    grid_search_pls_da = GridSearchCV(
        pipeline_pls_da, 
        param_grid_pls_da,
        cv=kf,                                                                             # same as earlier defines variable: stratified K fold cross-validation
        scoring=scoring,                                                                     # scoring is based on the earlier defines variable: ROC-AUC
        refit = True, 
        n_jobs=-1)                                                                      # use all avaiable CPU cores
    
    grid_search_pls_da.fit(X_train, y_train)
    classifier_PLS_DA = grid_search_pls_da.best_estimator_ 
    
    y_pred_pls_da = classifier_PLS_DA.predict(X_validate)
    probabilities_pls_da = classifier_PLS_DA.predict_proba(X_validate)

    print('Best parameters found:\n', grid_search_pls_da.best_params_)
    print("Beste score:", grid_search_pls_da.best_score_)
    print(f"CL Report of PLS-DA:\n", classification_report(
        y_validate, y_pred_pls_da, zero_division='warn'))
    AUC_plot_and_confusion_matrix(y_validate, probabilities_pls_da[:,1],
                                   y_validate, y_pred_pls_da, 
                                   "PLS DA model")

    # Pipeline Support Vector Machine--------------------------------------------------------------
    print('\n Start SVM pipeline', flush=True)
    pipeline_SVM = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('covariance_filter', DropCorrelatedFeatures(threshold=0.95)),
        ('classifier', SVC(
                        random_state=42,                                                # same seed for reproducibility
                        max_iter=10000,                                                 # define max iterations
                        class_weight='balanced', 
                        kernel = 'linear',
                        shrinking = True,
                        probability=True))])
    
    param_grid_SVM = {
        'selector': [RFE(SVC(                                                           # RFE as  estimator option
                            kernel='linear', 
                            max_iter=1000,                                              # define max iterations
                            random_state=42),                                           # same seed for reproducibility
                            n_features_to_select=15),                                   # select this amount of the most important features to keep
                    SFS(SVC(                                                            # SFS forward as estimator option
                            max_iter=1000,                                              # define max iterations
                            random_state=42),                                           # same seed for reproducibility
                            n_features_to_select="auto",                                # automatically determine the optimal number of features to keep
                            direction="forward",                                        # add the feature one by one till it doesn't improve
                            scoring=scoring,                                            # scoring is based on the earlier defines variable: ROC-AUC
                            cv=kf,                                                      # same as earlier defines variable: stratified K fold cross-validation
                            n_jobs=-1,                                                  # use all avaiable CPU cores
                            tol=1e-3),                                                  # stop if improvement is smaller than this value
                    SFS(SVC(                                                            # SFS backward as estimator option
                            max_iter=1000,                                              # define max iterations
                            random_state=42),                                           # same seed for reproducibility
                            n_features_to_select="auto",                                # automatically determine the optimal number of features to keep
                            direction="backward",                                       # remove a feature one by one till performance worsen beyond a threshold
                            scoring=scoring,                                            # scoring is based on the earlier defines variable: ROC-AUC
                            cv=kf,                                                      # same as earlier defines variable: stratified K fold cross-validation
                            n_jobs=-1,                                                  # use all avaiable CPU cores
                            tol=1e-3)],                                                 # stop if improvement is smaller than this value
        'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'classifier__C': [0.0001, 0.001, 0.01, 1, 5, 10, 100, 1000],
        'classifier__gamma':['auto', 'scale', 0.0001, 0.001, 0.01, 1, 10, 100, 1000]}

    grid_search_SVM = GridSearchCV(
        pipeline_SVM, 
        param_grid_SVM,
        cv=kf,                                                                           # same as earlier defines variable: stratified K fold cross-validation
        scoring=scoring,                                                                 # scoring is based on the earlier defines variable: ROC-AUC
        refit = True, 
        n_jobs=-1)                                                                       # use all avaiable CPU cores
   
    grid_search_SVM.fit(X_train, y_train)
    classifier_SVM = grid_search_SVM.best_estimator_ 

    y_pred_SVM = classifier_SVM.predict(X_validate) 
    probabilities_SVM = classifier_SVM.predict_proba(X_validate)

    print('Best parameters found:\n', grid_search_SVM.best_params_)
    print("Beste score:", grid_search_SVM.best_score_)
    print(f"CL Report of SVM:\n", classification_report(
        y_validate, y_pred_SVM, zero_division='warn'))
    AUC_plot_and_confusion_matrix(y_validate, probabilities_SVM[:,1], 
                                  y_validate, y_pred_SVM, 
                                  "Support vector machine")
    
    # Pipeline Gradient Boosting--------------------------------------------------------------
    pipeline_XGB = Pipeline(steps=[
        ('covariance_filter', DropCorrelatedFeatures(threshold=0.95)),
        ('classifier', xgb.XGBClassifier(
                        random_state=42,                                               # same seed for reproducibility
                        n_estimators=1000,
                        max_depth=10,
                        learning_rate=0.1
                        ))])
    
    param_grid_XGB = {
    'classifier__n_estimators': [100, 300, 500],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0]}

    grid_search_XGB = GridSearchCV(
        pipeline_XGB, 
        param_grid_XGB,
        cv=kf, 
        scoring=scoring,                                                             # scoring is based on the earlier defines variable: ROC-AUC
        refit = True, 
        n_jobs=-1)                                                                   # use all avaiable CPU cores

    grid_search_XGB.fit(X_train, y_train)
    classifier_XGB = grid_search_XGB.best_estimator_ 

    y_pred_XGB = classifier_XGB.predict(X_validate)  
    probabilities_XGB = classifier_XGB.predict_proba(X_validate)[:, 1]

    print('Best parameters found:\n', grid_search_XGB.best_params_)
    print("Beste score:", grid_search_XGB.best_score_)
    print(f"CL Report of XGB:\n", classification_report(
        y_validate, y_pred_XGB, zero_division='warn'))
    AUC_plot_and_confusion_matrix(y_validate, probabilities_XGB[:,1], 
                                  y_validate, y_pred_XGB, 
                                  "XGBoost model")

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
X_test_selected_LR = X_test[:, LR_selector]
X_test_selected_SVM = X_test[:, SVM_selector]

print("Test shape:", X_test.shape)
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
y_pred_pls_da = classifier_PLS_DA.predict(X_test)
probabilities_pls_da = classifier_PLS_DA.predict_proba(X_test)

print(f"CL Report of PLS-DA:\n", classification_report(y_test, y_pred_pls_da, zero_division='warn'))
AUC_plot_and_confusion_matrix(y_test, probabilities_pls_da[:,1], y_test, y_pred_pls_da, "PLS DA model", test=True)

# SVM test
y_pred_SVM = classifier_SVM.predict(X_test_selected_SVM) 
probabilities_SVM = classifier_SVM.predict_proba(X_test_selected_SVM)

print(f"CL Report of SVM:\n", classification_report(y_test, y_pred_SVM, zero_division='warn'))
AUC_plot_and_confusion_matrix(y_test, probabilities_SVM[:,1], y_test, y_pred_SVM, "Support vector machine", test=True)
    
# XGB test
y_pred_XGB = classifier_XGB.predict(X_test)  
propabilities_XGB = classifier_XGB.predict_proba(X_test)[:, 1]
if propabilities_XGB.ndim == 1:
    propabilities_XGB = np.column_stack([1 - propabilities_XGB, propabilities_XGB])

print(f"CL Report of XGB:\n", classification_report(y_test, y_pred_XGB, zero_division='warn'))
AUC_plot_and_confusion_matrix(y_test, propabilities_XGB[:,1], y_test, y_pred_XGB, "XGBoost model", test=True)


# %%