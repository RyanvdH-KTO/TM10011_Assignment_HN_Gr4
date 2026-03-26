#%% Import packages
# Import packages
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from feature_engine.selection import DropCorrelatedFeatures
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from functions import check_missing_values, split_features_target
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
    scoring = "roc_auc"
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

    # Define k-fold
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    #--------------------------------------------------------------
    # Pipeline PLS-DA
    pipeline_pls_da = Pipeline([
        ('scaler', MinMaxScaler()),
        ('covariance_filter', DropCorrelatedFeatures(threshold=0.95)),
        ('classifier', PLSRegression(n_components=10, random_state=42))
                        ])

    param_grid_pls_da = {
        'classifier__n_components': [5, 10, 15]
    }

    grid_search_pls_da = GridSearchCV(pipeline_pls_da, param_grid_pls_da, 
                                    cv=kf, scoring=scoring, refit = True, n_jobs=-1)
    
    grid_search_pls_da.fit(X_train, y_train)
    classifier_PLS_DA = grid_search_pls_da.best_estimator_ 
    
    y_pred_pls_da = classifier_PLS_DA.predict(X_validate)
    probabilities_pls_da = classifier_PLS_DA.predict_proba(X_validate)

    print('Best parameters found:\n', grid_search_pls_da.best_params_)
    print("Beste score:", grid_search_pls_da.best_score_)
    print(f"CL Report of PLS-DA:\n", classification_report(y_validate, y_pred_pls_da, zero_division='warn'))
    AUC_plot_and_confusion_matrix(y_validate, probabilities_pls_da[:,1], y_validate, y_pred_pls_da, "PLS DA model")

    return X_train, classifier_PLS_DA

#%% Run model
# Run model
if __name__ == "__main__":
    X_train, classifier_PLS_DA = main()
#%% Test with Test Data
# Load Test Data
test_data = pd.read_csv('hn/Test_data.csv', index_col=0)
print(f'The number of samples: {len(test_data.index)}')
print(f'The number of columns: {len(test_data.columns)}')
print(test_data['label'].value_counts())

# Preprocessing
check_missing_values(test_data)
X_test, y_test = split_features_target(test_data)

print("Test shape:", X_test.shape)
print("Test LR shape", X_test_selected_LR.shape)
print("Label distribution training set:\n", y_test.value_counts())

#%%
# PLS-DA test
y_pred_pls_da = classifier_PLS_DA.predict(X_test)
probabilities_pls_da = classifier_PLS_DA.predict_proba(X_test)

print(f"CL Report of PLS-DA:\n", classification_report(y_test, y_pred_pls_da, zero_division='warn'))
AUC_plot_and_confusion_matrix(y_test, probabilities_pls_da[:,1], y_test, y_pred_pls_da, "PLS DA model", test=True)


# %%