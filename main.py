# Load Data
from hn.load_data import load_data
data = load_data()
print(f'The number of samples: {len(data.index)}')

print(f'The number of columns: {len(data.columns)}')