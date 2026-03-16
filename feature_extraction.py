#%% Packages
# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

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

# Plot how many features survive at each threshold
plt.figure(figsize=(8, 4))
plt.plot(thresholds, n_features_kept, marker='o')
plt.axhline(y=features.shape[0], color='red', linestyle='--', label='n_samples (danger zone below this!)')
plt.xlabel("Variance threshold")
plt.ylabel("Number of features kept")
plt.title("Low Variance Filtering — how many features survive?")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

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
