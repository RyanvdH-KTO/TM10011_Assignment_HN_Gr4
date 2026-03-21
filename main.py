#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hn.load_data import load_data
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from functions import check_missing_values, split_features_target, scale_features, select_k_best_anova, rfe_selection, sfs_selection, pca_selection, remove_correlated_features
from functions import plot_correlation_matrix

#%% Load Data
# Load Data
data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

#%%
def preprocessing():
    

    # Check missing values
    check_missing_values(data)

    #Split dataset into features and encoded labels
    X, y = split_features_target(data)

    #Split into train and testset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    #Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("Train shape:", X_train_scaled.shape)
    print("Test shape:", X_test_scaled.shape)
    print("Label distribution training set:\n", y_train.value_counts())

    #Covariance feature elimination
    X_train_filtered, X_test_filtered, to_drop, surviving_cols = remove_correlated_features(X_train_scaled, X_test_scaled)

<<<<<<< HEAD
    return X_train_filtered

preprocessor = StandardScaler()
#preprocessor = X_train, y_train
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

pipeline_regression = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(penalty='l1', solver='saga', class_weight='balanced', random_state=42, max_iter=10000))
])

param_grid_regression = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10],
    'classifier__penalty': ['l1', 'l2', 'elasticnet']
}

grid_search_regression = GridSearchCV(pipeline_regression, param_grid_regression, cv=kf, scoring=["accuracy", "roc_auc", "f1"], refit = 'roc_auc', n_jobs=-1)
grid_search_regression.fit(X_train, y_train)

print('Best parameters found:\n', grid_search_regression.best_params_)
print("Beste score:", grid_search_regression.best_score_)
regression_model = grid_search_regression.best_estimator_ 
y_pred_regression = regression_model.predict(X_test) 
print(f"CL Report of PLS-DA:", classification_report(y_test, y_pred_regression, zero_division='warn'))
 
probabilities_regression = regression_model.predict_proba(X_test)

plot_auc(y_test, probabilities_regression[:,1])





#%%

    #Pipeline that compares feature selectors and classifiers
    feature_selectors = {
        "k best ANOVA" : SelectKBest(score_func=f_classif, k=10),
        "RFE"          : RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=5),
    }
=======
    #Plot the correlation matrix
    # plot_correlation_matrix(X_train_scaled, to_drop, feature_names=X.columns.tolist())
>>>>>>> 6e98c5cb80ecc3574f612701de4b7a1aa736891e

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
    X_test_filtered,
    y_train,
    k=10
    )

    # print("Shape train after k best ANOVA:", X_train_anova.shape)
    # print("Shape test after ANOVA:", X_test_anova.shape)


    # SFS forward
    X_train_sfs_fwd, X_test_sfs_fwd = sfs_selection(
    X_train_filtered,
    X_test_filtered,
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
        X_test_filtered,
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
        X_test_filtered,
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


# %%
