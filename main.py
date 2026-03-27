#%% Import packages
# Import packages
import numpy as np                                                                         # inport package for nummeric calculations and array operations 
import pandas as pd                                                                        # import package reading and handling tabular data
import xgboost as xgb                                                                      # import package for XGBoost classification
from sklearn.svm import SVC                                                                # import package fpr support vector machine classification
from sklearn.pipeline import Pipeline                                                      # import package for building preporcessing and modeling pipelines
from sklearn.feature_selection import RFE                                                  # import package for recursive feature elimination
from sklearn.preprocessing import MinMaxScaler                                             # import package scaling features to a fixed range
from sklearn.linear_model import LogisticRegression                                        # import package for logistic regression classsification
from sklearn.cross_decomposition import PLSRegression                                      # import package for partial least squares modeling
from sklearn.preprocessing import FunctionTransformer                                      # import package for applying custom transformations in pipelines
from sklearn.metrics import classification_report, roc_curve, auc                          # import package for evaluating classification models
from feature_engine.selection import DropCorrelatedFeatures                                # import package for removing highly correlated features
from sklearn.feature_selection import SequentialFeatureSelector as SFS                     # import package for sequential feature selection
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV        # import package for data splitting, cross-validation and grid search
from functions import check_missing_values, split_features_target                          # import zelf gebouwde functie
from functions import AUC_plot_and_confusion_matrix, ROC_STD_plot                          # import zelf gebouwde functie

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

    # Inner CV for GridSearch
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)                  # define k-fold for grid search
    # Outer CV for ROC ± std
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)                  # define k-fold for ROC

    #--------------------------------------------------------------
    # Pipeline Logistic regression
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
                                            n_features_to_select=25),                      # select this amount of the most important features to keep
                    SFS(LogisticRegression(                                                # SFS forward as feature selector option 
                                            max_iter=1000,                                 # define max iterations
                                            random_state=42),                              # same seed for reproducibility
                                            n_features_to_select="auto",                   # automatically determine the optimal number of features to keep
                                            direction="forward",                           # add the feature one by one till performance doesn't improve
                                            scoring=scoring,                               # scoring is based on the earlier defines variable: ROC-AUC
                                            cv=0,                                          # this is the cross-validation in the selector
                                            n_jobs=1),                                     # use 1 CPU core, so it runs faster
                    SFS(LogisticRegression(                                                # SFS backward as feature selector option 
                                            max_iter=1000,                                 # define max iterations
                                            random_state=42),                              # same seed for reproducibility
                                            n_features_to_select="auto",                   # automatically determine the optimal number of features to keep
                                            direction="backward",                          # remove a feature one by one till performance worsen beyond a threshold
                                            scoring=scoring,                               # scoring is based on the earlier defines variable: ROC-AUC
                                            cv=0,                                          # this is the cross-validation in the selector
                                            n_jobs=1)],                                    # use 1 CPU core, so it runs faster
        'classifier__C': [0.001, 0.01, 0.1, 1, 10],                                        # test different regularization strengths
        'classifier__penalty': ['l1', 'l2'],                                               # test L1 and L2 regularization as penalty
        'classifier__solver': ['liblinear']                                                # use liblinear as solver
    }, {                                                                                   # choose between these options, with different in penaly and solvers
        'selector': [RFE(LogisticRegression(                                               # RFE as  estimator option
                                            max_iter=1000,                                 # define max iterations
                                            random_state=42),                              # same seed for reproducibility
                                            n_features_to_select=25),                      # select this amount of the most important features to keep
                    SFS(LogisticRegression(                                                # SFS forward as feature selector option 
                                            max_iter=1000,                                 # define max iterations
                                            random_state=42),                              # same seed for reproducibility
                                            n_features_to_select="auto",                   # automatically determine the optimal number of features to keep
                                            direction="forward",                           # add the feature one by one till performance doesn't improve
                                            scoring=scoring,                               # scoring is based on the earlier defines variable: ROC-AUC
                                            cv=0,                                          # this is the cross-validation in the selector
                                            n_jobs=1),                                     # use 1 CPU core, so it runs faster
                    SFS(LogisticRegression(                                                # SFS backward as feature selector option 
                                            max_iter=1000,                                 # define max iterations
                                            random_state=42),                              # same seed for reproducibility
                                            n_features_to_select="auto",                   # automatically determine the optimal number of features to keep
                                            direction="backward",                          # remove a feature one by one till performance worsen beyond a threshold
                                            scoring=scoring,                               # scoring is based on the earlier defines variable: ROC-AUC
                                            cv=0,                                          # this is the cross-validation in the selector
                                            n_jobs=1)],                                    # use 1 CPU core, so it runs faster

        'classifier__C': [0.001, 0.01, 0.1, 1, 10],                                        # test different regularization strengths
        'classifier__penalty': ['elasticnet'],                                             # use elasticnet als penalty
        'classifier__solver': ['saga']}]                                                   # use saga as solver
   
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

    final_grid_search_regression = GridSearchCV(                                           # search the best parameters combination
        pipeline_regression,                                                               # for this model
        param_grid_regression,                                                             # with these parameters as option
        cv=inner_cv,                                                                       # same as earlier defines variable: stratified K fold cross-validation
        scoring=scoring,                                                                   # scoring is based on the earlier defines variable: ROC-AUC
        refit = True,                                                                      # retrain the best model on the full training set
        n_jobs=-1)                                                                         # use all avaiable CPU cores

    final_grid_search_regression.fit(X_train, y_train)                                     # fit the grid search on the training data
    classifier_LR = final_grid_search_regression.best_estimator_                           # store the best model

    y_pred_regression = classifier_LR.predict(X_validate)                                  # predicht the class labels for the validation set
    probabilities_regression = classifier_LR.predict_proba(X_validate)[:, 1]               # predicht the probabilities for the positive classes
    
    print('Best parameters found:\n', final_grid_search_regression.best_params_)           # print the best parameter combination
    print(f"CL Report of LR:\n", classification_report(                                    # print the classification metrics
        y_validate, y_pred_regression, zero_division='warn'))                              # compare true and predicted labels
    AUC_plot_and_confusion_matrix(y_validate, probabilities_regression,                    # plot the ROC-AUC curve and cofusion matrix
                                  y_validate, y_pred_regression,                           # use true matrix labels and predicted labels
                                  "Logistic regression model")                             # set the title of the plot
    ROC_STD_plot(mean_fpr, mean_tpr, mean_auc, std_auc, std_tpr)
    
    # Pipeline PLS-DA --------------------------------------------------------------------
    def squeeze_output(X):                                                                 # define a definition
        if isinstance(X, tuple):                                                           # if the output is a tuple
            X = X[0]                                                                       # if correct, than only keep the feature array
        return X.reshape(X.shape[0], -1)                                                   # reshape the output into a 2D array with samples as rows
    
    pipeline_pls_da = Pipeline([                                                           # define the pipeline
        ('scaler', MinMaxScaler()),                                                        # scale features by minmaxscaler method
        ('covariance_filter', DropCorrelatedFeatures(threshold=0.95)),                     # filter for covariance, drop features which are correlated higher than 0.95
        ('pls', PLSRegression(                                                             # transform the data into a smaller set of latent PLS components
            n_components=10,                                                               # use 10 components to summarize the most relevant information
            scale=False,                                                                   # disable internal scaling because scaling is already done earlier
            max_iter=1000)),                                                               # define max iterations
        ('squeeze', FunctionTransformer(squeeze_output)),                                  # reshape the PLS output into the correct 2D format for the classifier
        ('classifier', LogisticRegression(                                                 # define the final Logistic Regression classifier
                        penalty='elasticnet',                                              # use elasticnet penalty
                        solver='saga',                                                     # use the solver that supports Elastic Net regularization
                        class_weight='balanced',                                           # automatically give more weight to the minority class
                        l1_ratio=0.85,                                                     # set the balance between L1 and L2 regularization
                        random_state=42,                                                   # same seed for reproducibility
                        max_iter=1000))])                                                  # define max iterations

    param_grid_pls_da = {                                                                  # define the grid-parameter
        'pls__n_components': [5, 10, 15],                                                  # test different numbers of PLS components
        'classifier__C': [0.001, 0.01, 0.1, 1, 10]}                                        # test different regularization strengths

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train_idx, val_idx in outer_cv.split(X_train,y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        grid_search_pls_da = GridSearchCV(pipeline_pls_da, param_grid_pls_da,
                                        cv=inner_cv, scoring=scoring, refit = True, n_jobs=-1)
        grid_search_pls_da.fit(X_tr, y_tr)
        best_model = grid_search_pls_da.best_estimator_

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

    final_grid_search_pls_da = GridSearchCV(                                               # search the best parameters combination
        pipeline_pls_da,                                                                   # for this model
        param_grid_pls_da,                                                                 # with these parameters as option
        cv=inner_cv,                                                                       # same as earlier defines variable: stratified K fold cross-validation
        scoring=scoring,                                                                   # scoring is based on the earlier defines variable: ROC-AUC
        refit = True,                                                                      # retrain the best model on the full training set
        n_jobs=-1)                                                                         # use all avaiable CPU cores
    
    final_grid_search_pls_da.fit(X_train, y_train)                                         # fit the grid search on the training data
    classifier_PLS_DA = final_grid_search_pls_da.best_estimator_                           # store the best model

    y_pred_pls_da = classifier_PLS_DA.predict(X_validate)                                  # predict the class labels for the validation set
    probabilities_pls_da = classifier_PLS_DA.predict_proba(X_validate)                     # predict the probabilities for the positive classes

    print('Best parameters found:\n', final_grid_search_pls_da.best_params_)               # print the best parameter combination
    print(f"CL Report of PLS-DA:\n", classification_report(                                # print the classification metrics
        y_validate, y_pred_pls_da, zero_division='warn'))                                  # compare true and predicted labels
    AUC_plot_and_confusion_matrix(y_validate, probabilities_pls_da[:,1],                   # plot the ROC-AUC curve and cofusion matrix
                                   y_validate, y_pred_pls_da,                              # use true matrix labels and predicted labels
                                   "PLS DA model")                                         # set the title of the plot
    ROC_STD_plot(mean_fpr, mean_tpr, mean_auc, std_auc, std_tpr)

    #--------------------------------------------------------------
    # Pipeline Support Vector Machine
    pipeline_SVM = Pipeline(steps=[                                                        # define the pipeline
        ('scaler', MinMaxScaler()),                                                        # scale features by minmaxscaler method
        ('covariance_filter', DropCorrelatedFeatures(threshold=0.95)),                     # filter for covariance, drop features which are correlated higher than 0.95
        ('classifier', SVC(                                                                # define classifier
                        random_state=42,                                                   # same seed for reproducibility
                        max_iter=10000,                                                    # define max iterations
                        class_weight='balanced',                                           # give more weight to the minority class
                        kernel = 'linear',                                                 # use this kernel as default, meaning the model separates classes with a straight decision boundary 
                        shrinking = True,                                                  # use a faster optimization shortcut by temporarily ignoring less important support vectors, makes it much faster and does not really worsen the performance
                        probability=True))])                                               # enable probability estimates for each prediction

    param_grid_SVM = {                                                                     # define the grid-parameter
        'selector': [RFE(SVC(                                                              # RFE as  estimator option
                            kernel='linear',                                               #  use a linear kernel so feature importance can be derived from the model
                            max_iter=1000,                                                 # define max iterations
                            random_state=42),                                              # same seed for reproducibility
                            n_features_to_select=15),                                      # select this amount of the most important features to keep
                    SFS(SVC(                                                               # SFS forward as feature selector option 
                            max_iter=1000,                                                 # define max iterations
                            random_state=42),                                              # same seed for reproducibility
                            n_features_to_select="auto",                                   # automatically determine the optimal number of features to keep
                            direction="forward",                                           # add the feature one by one till it doesn't improve
                            scoring=scoring,                                               # scoring is based on the earlier defines variable: ROC-AUC
                            cv=0,                                                          # this is the cross-validation in the selector
                            n_jobs=1),                                                     # use 1 CPU core, so it runs faster
                    SFS(SVC(                                                               # SFS backward as feature selector option 
                            max_iter=1000,                                                 # define max iterations
                            random_state=42),                                              # same seed for reproducibility
                            n_features_to_select="auto",                                   # automatically determine the optimal number of features to keep
                            direction="backward",                                          # remove a feature one by one till performance worsen beyond a threshold
                            scoring=scoring,                                               # scoring is based on the earlier defines variable: ROC-AUC
                            cv=0,                                                          # this is the cross-validation in the selector
                            n_jobs=1)],                                                    # use 1 CPU core, so it runs faster
        'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],                        # test different kernel functions to learn linear or non-linear decision boundaries
        'classifier__C': [0.0001, 0.001, 0.01, 1, 5, 10, 100, 1000],                       # test different regularization strengths
        'classifier__gamma':['auto', 'scale', 0.0001, 0.001, 0.01, 1, 10, 100, 1000]}      # test different gamma values, which control how far the influence of one sample reaches

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train_idx, val_idx in outer_cv.split(X_train,y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        grid_search_SVM = GridSearchCV(pipeline_SVM, param_grid_SVM,
                                       cv=inner_cv, scoring=scoring, refit = True, n_jobs=-1)
        grid_search_SVM.fit(X_tr, y_tr)
        best_model = grid_search_SVM.best_estimator_

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

    final_grid_search_SVM = GridSearchCV(                                                  # search the best parameters combination
        pipeline_SVM,                                                                      # for this model
        param_grid_SVM,                                                                    # with these parameters as option
        cv=inner_cv,                                                                       # same as earlier defines variable: stratified K fold cross-validation
        scoring=scoring,                                                                   # scoring is based on the earlier defines variable: ROC-AUC
        refit = True,                                                                      # retrain the best model on the full training set
        n_jobs=-1)                                                                         # use all avaiable CPU cores
   
    final_grid_search_SVM.fit(X_train, y_train)                                            # fit the grid search on the training data
    classifier_SVM = final_grid_search_SVM.best_estimator_                                 # store the best model

    y_pred_SVM = classifier_SVM.predict(X_validate)                                        # predicht the class labels for the validation set
    probabilities_SVM = classifier_SVM.predict_proba(X_validate)                           # predicht the probabilities for the positive classes

    print('Best parameters found:\n', final_grid_search_SVM.best_params_)                  # print the best parameter combination
    print(f"CL Report of SVM:\n", classification_report(                                   # print the classification metrics
        y_validate, y_pred_SVM, zero_division='warn'))                                     # compare true and predicted labels
    AUC_plot_and_confusion_matrix(y_validate, probabilities_SVM[:,1],                      # plot the ROC-AUC curve and cofusion matrix
                                  y_validate, y_pred_SVM,                                  # use true matrix labels and predicted labels
                                  "Support vector machine")                                # set the title of the plot
    ROC_STD_plot(mean_fpr, mean_tpr, mean_auc, std_auc, std_tpr)

    #--------------------------------------------------------------
    # Pipeline Gradient Boosting
    pipeline_XGB = Pipeline(steps=[                                                        # define the pipeline
        ('covariance_filter', DropCorrelatedFeatures(threshold=0.95)),                     # filter for covariance, drop features which are correlated higher than 0.95
        ('classifier', xgb.XGBClassifier(                                                  # define classifier
                        random_state=42,                                                   # same seed for reproducibility
                        n_estimators=1000,                                                 # use 1000 boosting trees as the initial setting
                        max_depth=10,                                                      # set the maximum depth of each tree
                        learning_rate=0.1))])                                              # set the step size used to update the model during training
    
    param_grid_XGB = {                                                                     # define the grid-parameter
    'classifier__n_estimators': [100, 300, 500],                                           # test different numbers of trees
    'classifier__max_depth': [3, 5, 7],                                                    # test different numbers of three depths to control model complexicity
    'classifier__learning_rate': [0.01, 0.05, 0.1],                                        # test different learning rates for the boosting process
    'classifier__subsample': [0.8, 1.0],                                                   # test different fractions of training samples used per tree
    'classifier__colsample_bytree': [0.8, 1.0]}                                            # test different fractions of features used per tree

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train_idx, val_idx in outer_cv.split(X_train,y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        grid_search_XGB = GridSearchCV(pipeline_XGB, param_grid_XGB,
                                        cv=inner_cv, scoring=scoring, refit = True, n_jobs=-1)
        grid_search_XGB.fit(X_tr, y_tr)
        best_model = grid_search_XGB.best_estimator_

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

    final_grid_search_regression = GridSearchCV(                                           # search the best parameters combination
        pipeline_regression,                                                               # for this model
        param_grid_regression,                                                             # with these parameters as option
        cv=inner_cv,                                                                       # same as earlier defines variable: stratified K fold cross-validation
        scoring=scoring,                                                                   # scoring is based on the earlier defines variable: ROC-AUC
        refit = True,                                                                      # retrain the best model on the full training set
        n_jobs=-1)                                                                         # use all avaiable CPU cores

    final_grid_search_regression.fit(X_train, y_train)                                     # fit the grid search on the training data
    classifier_LR = final_grid_search_regression.best_estimator_                           # store the best model

    y_pred_regression = classifier_LR.predict(X_validate)                                  # predicht the class labels for the validation set
    probabilities_regression = classifier_LR.predict_proba(X_validate)[:, 1]               # predicht the probabilities for the positive classes
    
    print('Best parameters found:\n', final_grid_search_regression.best_params_)           # print the best parameter combination
    print("Beste score:", final_grid_search_regression.best_score_)                        # print the best cross-validation score
    print(f"CL Report of LR:\n", classification_report(                                    # print the classification metrics
        y_validate, y_pred_regression, zero_division='warn'))                              # compare true and predicted labels
    AUC_plot_and_confusion_matrix(y_validate, probabilities_regression,                    # plot the ROC-AUC curve and cofusion matrix
                                  y_validate, y_pred_regression,                           # use true matrix labels and predicted labels
                                  "Logistic regression model")                             # set the title of the plot
    ROC_STD_plot(mean_fpr, mean_tpr, mean_auc, std_auc, std_tpr)

    _final_grid_search_XGB = GridSearchCV(                                                 # search the best parameters combination
        pipeline_XGB,                                                                      # for this model
        param_grid_XGB,                                                                    # with these parameters as option
        cv=inner_cv,                                                                       # same as earlier defines variable: stratified K fold cross-validation
        scoring=scoring,                                                                   # scoring is based on the earlier defines variable: ROC-AUC
        refit = True,                                                                      # retrain the best model on the full training set
        n_jobs=-1)                                                                         # use all avaiable CPU cores

    _final_grid_search_XGB.fit(X_train, y_train)                                           # fit the grid search on the training data
    classifier_XGB = _final_grid_search_XGB.best_estimator_                                # store the best model

    y_pred_XGB = classifier_XGB.predict(X_validate)                                        # predicht the class labels for the validation set
    probabilities_XGB = classifier_XGB.predict_proba(X_validate)[:, 1]                     # predicht the probabilities for the positive classes

    print('Best parameters found:\n', _final_grid_search_XGB.best_params_)                 # print the best parameter combination
    print(f"CL Report of XGB:\n", classification_report(                                   # print the classification metrics
        y_validate, y_pred_XGB, zero_division='warn'))                                     # compare true and predicted labels
    AUC_plot_and_confusion_matrix(y_validate, probabilities_XGB[:,1],                      # plot the ROC-AUC curve and cofusion matrix
                                  y_validate, y_pred_XGB,                                  # use true matrix labels and predicted labels
                                  "XGBoost model")                                         # set the title of the plot
    ROC_STD_plot(mean_fpr, mean_tpr, mean_auc, std_auc, std_tpr)

    return X_train, classifier_LR, classifier_PLS_DA, classifier_SVM, classifier_XGB       # end the definition and give back these data and models

#%% Run model
if __name__ == "__main__":                                                                 # run this code only when the script is executed directly
    X_train, classifier_LR, classifier_PLS_DA, classifier_SVM, classifier_XGB, LR_selector, SVM_selector = main()  # run the main function and store the trained models and selected features

#%% Test with Test Data
test_data = pd.read_csv('hn/Test_data.csv', index_col=0)                                   # load the test data by reading the CSV
print(f'The number of samples: {len(test_data.index)}')                                    # print the number of samples. The number of rows equal the number of samples
print(f'The number of features: {len(test_data.columns)}')                                 # print the number of features/ The number of columns correspond to the number of features
print(test_data['label'].value_counts())                                                   # print datatype

check_missing_values(test_data)                                                            # check for missing values
X_test, y_test = split_features_target(test_data)                                          # split the test dataset into features and target labels

print("Test shape:", X_test.shape)                                                         # print the shape of the full test feature set
print("Label distribution training set:\n", y_test.value_counts())                         # print the class distribution of the test target labels

#%% LR test
y_pred_regression = classifier_LR.predict(X_test)                                          # predict de labels
probabilities_regression = classifier_LR.predict_proba(X_test)[:, 1]                       # predict the probabilities
print(y_pred_regression.shape)                                                             # print the shape of the predicted labels
print(f"CL Report of LR:\n", classification_report(y_test, y_pred_regression, zero_division='warn'))      # print the classification metrics
AUC_plot_and_confusion_matrix(y_test, probabilities_regression,                            # plot the ROC-AUC curve and cofusion matrix
                              y_test, y_pred_regression,                                   # use true matrix labels and predicted labels
                              "Logistic regression model",                                 # set the title of the plot and confusion matrix
                              test=True)                                                   # put the test setting on on

# PLS-DA test
y_pred_pls_da = classifier_PLS_DA.predict(X_test)
probabilities_pls_da = classifier_PLS_DA.predict_proba(X_test)

print(f"CL Report of PLS-DA:\n", classification_report(y_test, y_pred_pls_da, zero_division='warn'))
AUC_plot_and_confusion_matrix(y_test, probabilities_pls_da[:,1], y_test, y_pred_pls_da, "PLS DA model", test=True)

# SVM test
y_pred_SVM = classifier_SVM.predict(X_test) 
probabilities_SVM = classifier_SVM.predict_proba(X_test)

print(f"CL Report of SVM:\n", classification_report(y_test, y_pred_SVM, zero_division='warn'))
AUC_plot_and_confusion_matrix(y_test, probabilities_SVM, y_test, y_pred_SVM, "Support vector machine", test=True)
    
# XGB test
y_pred_XGB = classifier_XGB.predict(X_test)  
propabilities_XGB = classifier_XGB.predict_proba(X_test)

print(f"CL Report of XGB:\n", classification_report(y_test, y_pred_XGB, zero_division='warn'))
AUC_plot_and_confusion_matrix(y_test, propabilities_XGB[:,1], y_test, y_pred_XGB, "XGBoost model", test=True)

# %%
