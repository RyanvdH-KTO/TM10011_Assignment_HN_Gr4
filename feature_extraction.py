#%% Packages
# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector

#%% Import and select data
#Import and select data
from hn.load_data import load_data
data = load_data()

# Separate features from label
features = data.drop(columns=["label"])  # everything except the label
target_vector = data["label"]            # just the label column

print(target_vector)

#%% Scalar
#Scale the features 
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
le = LabelEncoder()
y = le.fit_transform(target_vector) 

print(features_scaled)
print(y)

# %% Variance Filtering
# Try a range of thresholds
thresholds = np.arange(0.0, 1.0, 0.01)
n_features_kept = []

for t in thresholds:
    selector = VarianceThreshold(threshold=t)
    selector.fit(features)
    n_features_kept.append(selector.get_support().sum())

selector = VarianceThreshold(threshold=0.01)
features_filtered = selector.fit_transform(features)  

print(f"Features before : {features.shape[1]}")
print(f"Features after  : {features_filtered.shape[1]}")

# Which features survived? 
surviving_feature_names = features.columns[selector.get_support()].tolist()
print(f"\nSurviving features:\n{surviving_feature_names}")

# Count surviving features per group
groups = ["sf", "hf", "tf", "of"]

for group in groups:
    # count how many surviving feature names start with this prefix
    count = sum(1 for name in surviving_feature_names if name.startswith(group))
    total = sum(1 for name in features.columns if name.startswith(group))
    print(f"{group}: {count} / {total} survived")




# %%
# Univariate feature selection: kiest de beste k features op basis van hun individuele relatie met de target
def select_k_best_anova(X_train, X_test, y_train, k=10): #k zegt hoeveel features je wil
    selector = SelectKBest(score_func=f_classif, k=k) #SelectKBest: kijkt naar alle features afzonderlijk, geeft elke feature een score, en kiest daarna de k hoogste scores
                                                      # f_classif: betekent dat je de ANOVA F-test gebruikt
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    selected_indices = selector.get_support(indices=True) #welke features gekozen zijn
    scores = selector.scores_ #ANOVA scores per feature

    print("Selected feature indices:", selected_indices)
    print("Scores of all features:", scores)
    print("Shape train after selection:", X_train_sel.shape)
    print("Shape test after selection:", X_test_sel.shape)

    return X_train_sel, X_test_sel, selected_indices, scores, selector


from preprocessing import preprocess_pipeline
X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(scale_method="standard")

X_train_sel, X_test_sel, selected_indices, scores, selector = select_k_best_anova(
    X_train, X_test, y_train, k=10
)


# %%
# RFE: recursive feature elimination
def rfe_selection(X_train, X_test, y_train, estimator, n_features=10):

    selector = RFE(estimator=estimator, n_features_to_select=n_features)

    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    selected_indices = selector.get_support(indices=True)
    ranking = selector.ranking_

    print("Selected feature indices:", selected_indices)
    print("Ranking of all features:", ranking) #Ranking zegt iets over in welke ronde de feature eruit is gegooid
    print("Shape train after selection:", X_train_sel.shape)
    print("Shape test after selection:", X_test_sel.shape)

    return X_train_sel, X_test_sel, selected_indices, ranking, selector

# RFE testen met Logistic Regression
X_train_rfe, X_test_rfe, selected_idx, ranking, selector = rfe_selection(
    X_train,
    X_test,
    y_train,
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    n_features=10
)



# %%
# Sequential Feature Selector
from sklearn.feature_selection import SequentialFeatureSelector
def sfs_selection(X_train, X_test, y_train, estimator, n_features, tol, direction="forward", scoring="accuracy", cv=5):
    selector = SequentialFeatureSelector(
        estimator,
        n_features_to_select=n_features,
        direction=direction,
        scoring=scoring,
        cv=cv,
        n_jobs=-1
    )

    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    selected_indices = selector.get_support(indices=True)

    print("Selected feature indices:", selected_indices)
    print("Shape train after selection:", X_train_sel.shape)
    print("Shape test after selection:", X_test_sel.shape)

    return X_train_sel, X_test_sel, selected_indices, ranking, selector

# SFS testen met Logistic Regression
X_train_sfs, X_test_sfs, selected_idx, selector = sfs_selection(
    X_train,
    X_test,
    y_train,
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    n_features='auto',
    tol=0.01
)



from sklearn.feature_selection import SequentialFeatureSelector

def perform_sfs(features_train, label_train, model, n_splits=5):
    sfs = SequentialFeatureSelector(
        estimator=model,
        n_features_to_select='auto',
        tol=0.01,
        direction='forward',
        scoring='accuracy',
        cv=n_splits,
        n_jobs=-1
    )

    sfs.fit(features_train, label_train)

    selected_indices = sfs.get_support(indices=True)

    print("Selected feature indices:", selected_indices)
    print("Number of selected features:", len(selected_indices))

    return sfs, selected_indices
# %%



# %% PCA feature selection
#PCA feature selection pipeline
from sklearn import decomposition 
import seaborn

pca = decomposition.PCA(n_components=0.95)
pca.fit(features_scaled)
X_pca = pca.transform(features_scaled)

print(X_pca.shape)
print(pca.explained_variance_ratio_)
print("Variance:", pca.explained_variance_ratio_)
