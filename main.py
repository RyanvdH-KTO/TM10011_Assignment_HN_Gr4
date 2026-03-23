#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hn.load_data import load_data
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report
from functions import check_missing_values, split_features_target, scale_features, select_k_best_anova, rfe_selection, sfs_selection, pca_selection, remove_correlated_features
from functions import plot_correlation_matrix
from sklearn.preprocessing import FunctionTransformer
from sklearn.cross_decomposition import PLSRegression

#%% Load Data
# Load Data
data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

data, final_test = train_test_split(data, test_size=0.2, random_state=42)
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

#%% Def Plot AUC-curve
def plot_auc(labels, probs):
    # info regression
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(labels.values.ravel(), probs.ravel())
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color = 'blue', label = 'AUC: %0.3f' % roc_auc, linestyle='solid')
    plt.plot([0, 1], [0, 1], color = 'grey', linestyle=(0, (5, 10)), label='Random prediction')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC')
    plt.legend()
    plt.show()
    
#%%
def main():
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

    #Scale features
    X_train_scaled, X_validate_scaled, scaler = scale_features(X_train, X_validate)

    print("Train shape:", X_train_scaled.shape)
    print("Test shape:", X_validate_scaled.shape)
    print("Label distribution training set:\n", y_train.value_counts())

    #Covariance feature elimination
    X_train_filtered, X_validate_filtered, to_drop, surviving_cols = remove_correlated_features(X_train_scaled, X_validate_scaled)


    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Pipeline Logistic regression
    pipeline_regression = Pipeline(steps=[
        ('classifier', LogisticRegression(penalty='l1', solver='saga', class_weight='balanced', random_state=42, max_iter=10000))
    ])

    param_grid_regression = {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10],
        'classifier__penalty': ['l1', 'l2', 'elasticnet']
    }

    grid_search_regression = GridSearchCV(pipeline_regression, param_grid_regression, cv=kf, scoring=["accuracy", "roc_auc", "f1"], refit = 'roc_auc', n_jobs=-1)
    grid_search_regression.fit(X_train_filtered, y_train)

    regression_model = grid_search_regression.best_estimator_ 
    y_pred_regression = regression_model.predict(X_validate_filtered)
    probabilities_regression = regression_model.predict_proba(X_validate_filtered)

    print('Best parameters found:\n', grid_search_regression.best_params_)
    print("Beste score:", grid_search_regression.best_score_)
    print(f"CL Report of PLS-DA:", classification_report(y_validate, y_pred_regression, zero_division='warn'))
    plot_auc(y_validate, probabilities_regression[:,1])


    # Pipeline PLS-DA
    def squeeze_output(X):
        if isinstance(X, tuple):
            X = X[0]
        return X.reshape(X.shape[0], -1)

    pipeline_pls_da = Pipeline([
        ('pls', PLSRegression(n_components=1, scale=False, max_iter=10)),
        ('squeeze', FunctionTransformer(squeeze_output)),
        ('classifier', LogisticRegression(
            penalty='elasticnet', solver='saga', class_weight='balanced', 
            random_state=42, max_iter=10
        ))
    ])

    param_grid_pls_da = {
        'pls__n_components': [5, 10, 15],
        'classifier__C': [0.001, 0.01, 0.1, 1, 10]
    }

    grid_search_pls_da = GridSearchCV(pipeline_pls_da, param_grid_pls_da, 
                                    cv=kf, scoring=["accuracy", "roc_auc", "f1"], refit = 'roc_auc', n_jobs=-1)
    grid_search_pls_da.fit(X_train_filtered, y_train)

    pls_da_model = grid_search_pls_da.best_estimator_ 
    y_pred_pls_da = pls_da_model.predict(X_validate_filtered)
    probabilities_pls_da = pls_da_model.predict_proba(X_validate_filtered)

    print('Best parameters found:\n', grid_search_pls_da.best_params_)
    print("Beste score:", grid_search_pls_da.best_score_)
    print(f"CL Report of PLS-DA:", classification_report(y_validate, y_pred_pls_da, zero_division='warn'))
    plot_auc(y_validate, probabilities_pls_da[:,1])

    # (Lineair) Support Vector Machine
    pipeline_SVM = Pipeline(steps=[
    ('classifier', SVC(random_state=42, 
                     max_iter=5000, 
                     class_weight='balanced', 
                     kernel = 'linear'
                     )) 
                     ])
    
    param_grid_SVM = {
    'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'classifier__C': [0.0001, 0.001, 0.01, 1, 5, 10, 100, 1000],
    'classifier__gamma':['auto', 'scale', 0.0001, 0.001, 0.01, 1, 10, 100, 1000],
    'classifier__class_weight': ['none', 'balanced'],
    'classifier__shrinking':[True, False],
    'classifier__tol':[1, 0.1, 1e-2, 1e-3, 1e-4]}

    grid_search_SVM = GridSearchCV(
    pipeline_SVM,
    param_grid_SVM,
    cv=kf, 
    scoring=["accuracy", "roc_auc", "f1"],
    refit = "roc_auc",  
    n_jobs=-1,
    )

    grid_search_SVM.fit(X_train_filtered, y_train)
    classifier_SVM = grid_search_SVM.best_estimator_ 
    y_pred_SVM = classifier_SVM.predict(X_validate_filtered) 
    probabilities_SVM = classifier_SVM.predict_proba(X_validate_filtered)

    print('Best parameters found:\n', grid_search_SVM.best_params_)
    print("Beste score:", grid_search_SVM.best_score_)
    print(f"CL Report of PLS-DA:", classification_report(y_validate, y_pred_SVM, zero_division='warn'))
    plot_auc(y_validate, probabilities_SVM[:,1])





     #%Gradient Boosting
    pipeline_XGB = Pipeline(steps=[
        ('classifier', xgb.XGBClassifier(
        random_state=42,
        n_estimators=1000,
        max_depth=10,
        learning_rate=0.1)) 
        ])
    
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
    scoring=["accuracy", "roc_auc", "f1"], 
    refit = "roc_auc",
    n_jobs=-1,
    )

    grid_search_XGB.fit(X_train_filtered, y_train) 
    classifier_XGB = grid_search_XGB.best_estimator_ 
    y_pred_XGB = classifier_XGB.predict(X_validate_filtered)  
    propabilities_XGB = classifier_XGB.predict_proba(X_validate_filtered)[:, 1]

    print('Best parameters found:\n', grid_search_XGB.best_params_)
    print("Beste score:", grid_search_XGB.best_score_)
    print(f"CL Report of PLS-DA:", classification_report(y_validate, y_pred_XGB, zero_division='warn'))
    plot_auc(y_validate, propabilities_XGB[:,1])


#%%

    #Pipeline that compares feature selectors and classifiers
    feature_selectors = {
        "k best ANOVA" : SelectKBest(score_func=f_classif, k=10),
        "RFE"          : RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=5),
    }

    # Feature selection
    # Deze uitvoeren voor estimator = logistic regresion en voor de SVM. 

    # Logistic Regression estimator for feature selection
    lr_estimator = LogisticRegression(
        penalty='l2',
        solver='liblinear',
        class_weight='balanced',
        random_state=42,
        max_iter=10000
    )
    # Deze gebruikte ik even om te testen, maar hier moet dan goeie LR model of SVM

    # Feature selection
    # Select k best ANOVA
    X_train_anova, X_test_anova = select_k_best_anova(
    X_train_filtered,
    X_validate_filtered,
    y_train,
    k=10
    )

    # print("Shape train after k best ANOVA:", X_train_anova.shape)
    # print("Shape test after ANOVA:", X_test_anova.shape)


    # SFS forward
    X_train_sfs_fwd, X_test_sfs_fwd = sfs_selection(
    X_train_filtered,
    X_validate_filtered,
    y_train,
    estimator=lr_estimator,
    direction="forward",
    scoring="accuracy", #kan ook roc-auc
    cv=5
    ) 

    # print("Shape train after SFS forward:", X_train_sfs_fwd.shape)
    # print("Shape test after SFS forward:", X_test_sfs_fwd.shape)

    # SFS backward
    X_train_sfs_bwd, X_test_sfs_bwd = sfs_selection(
        X_train_filtered,
        X_validate_filtered,
        y_train,
        estimator=lr_estimator,
        direction="backward",
        scoring="accuracy",
        cv=5
    )  

    # print("Shape train after SFS backward:", X_train_sfs_bwd.shape)
    # print("Shape test after SFS backward:", X_test_sfs_bwd.shape)

    # RFE
    X_train_rfe, X_test_rfe = rfe_selection(
        X_train_filtered,
        X_validate_filtered,
        y_train,
        estimator=lr_estimator,
        n_features=10
    )

    # print("Shape train after RFE:", X_train_rfe.shape)
    # print("Shape test after RFE:", X_test_rfe.shape)

    # Logistic Regression Classifier
    '''Needs feature selection beforehand, so we compare different methods to be able to use the best in the pipeline'''

    #%% SVM classifier
    #SVM classifier
    '''Needs feature selection beforehand, so we compare different methods to be able to use the best in the pipeline'''


    #%% XGBoost Classifier
    #XGBoost Classifier
    '''Doesn't need further feature selection, since this method handles that itself'''

    #%% Partial Least Square classifer
    #PLS classifier
    '''Doesn't need further feature selection, since this method handles that itself'''


    #%% Classifier Evaluation 

#%%
if __name__ == "__main__":
    main()