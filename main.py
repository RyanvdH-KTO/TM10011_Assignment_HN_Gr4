#%% packages
# load packages
import pandas as pd

# load functions
from hn.load_data import load_data

#%% Data

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

data = pd.DataFrame(data)

# %%
