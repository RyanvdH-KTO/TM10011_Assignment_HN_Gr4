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
    X_train_filtered, X_test_filtered = remove_correlated_features(X_train_scaled, X_test_scaled)
    print(X_train_scaled)

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

    classifiers = {
        "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    }

    # results = []
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # for sel_name, selector in feature_selectors.items():
    #     for clf_name, clf in classifiers.items():
            
    #         selector.fit(X_train_scaled, y_train)
    #         selected_indices = selector.get_support(indices=True)
    #         selected_names = X.columns[selected_indices].tolist()
    #         print(f"\n[{sel_name}] Selected features: {selected_names}")

    #         pipe = Pipeline([
    #             ("selector", selector),
    #             ("clf",      clf),
    #         ])
            
    #         #Get different cross-validation scores
    #         scores = cross_validate(
    #             pipe, X_train_scaled, y_train,
    #             cv=cv,
    #             scoring=["accuracy", "roc_auc", "f1"], 
    #             n_jobs=-1
    #         )

    #         results.append({
    #             "Feature Selection" : sel_name,
    #             "Classifier"        : clf_name,
    #             "Accuracy"          : scores["test_accuracy"].mean(),
    #             "ROC-AUC"           : scores["test_roc_auc"].mean(),
    #             "F1"                : scores["test_f1"].mean(),
    #         })

    # results_df = pd.DataFrame(results)
    # print(results_df)
    # return results_df, X_test, y_test


#%%
if __name__ == "__main__":
    main()


# %%
