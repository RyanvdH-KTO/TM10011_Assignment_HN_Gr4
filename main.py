#%%
import pandas as pd
from hn.load_data import load_data
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from hn.load_data import load_data
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

#%% Load Data
# Load Data
# data = load_data()
# print(f'The number of samples: {len(data.index)}')
# print(f'The number of columns: {len(data.columns)}')

# data = pd.DataFrame(data)

#%%
def load_dataset():
    data = load_data()
    data = pd.DataFrame(data)

    print("Dataset shape:", data.shape)
    return data


# %%
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


# %%
# Data splitten in features (X) en target (y)
def split_features_target(data, label_col='label'):
    X = data.drop(columns=[label_col]) 
    y = data[label_col]
    return X, y


# %%
# Encode lables: T12 = 0, T34 = 1
def encode_labels(y):
    y_encoded = y.map({'T12': 0, 'T34': 1})
    return y_encoded


#%%
# Train test split (stratified)
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test

#%%
# Feature scaling
# Meerdere scalars vergelijken: kijken welke beste accuracy geeft (en dan bv standard als baseline gebruiken)
def scale_features(X_train, X_test, method="standard"):

    if method == "standard":
        scaler = StandardScaler()

    elif method == "minmax":
        scaler = MinMaxScaler()

    elif method == "robust":
        scaler = RobustScaler()

    else:
        raise ValueError("Unknown scaler")

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler



# %%

def preprocess_pipeline(scale_method="standard"):

    # Load dataset
    data = load_dataset()

    # Check missing values
    check_missing_values(data)

    # Split features and target
    X, y = split_features_target(data)

    print("Feature shape:", X.shape)
    print("Original label distribution:")
    print(y.value_counts())

    # Encode labels
    y = encode_labels(y)

    print("Encoded label distribution:")
    print(y.value_counts())

    # Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Train shape before scaling:", X_train.shape)
    print("Test shape before scaling:", X_test.shape)

    # Feature scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, method=scale_method
    )

    print("Final shape training:", X_train_scaled.shape)
    print("Final shape test:", X_test_scaled.shape)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


#%%
def main():
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(scale_method="standard")
    return X_train, X_test, y_train, y_test, scaler


#%%
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = main()

# %% Elastic-Net-Logistic-Regression
#Elastic-Net-Logistic-Regression

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
preprocessor = X_test, y_test

pipeline_regression = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(penalty='l1', solver='saga', class_weight='balanced', random_state=42, max_iter=10000))
])

param_grid_regression = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10]
}

grid_search_regression = GridSearchCV(pipeline_regression, param_grid_regression, cv=kf, scoring='roc_auc', n_jobs=-1)
grid_search_regression.fit(X_train, y_train)

print('Best parameters found:\n', grid_search_regression.best_params_)
print("Beste score:", grid_search_regression.best_score_)
