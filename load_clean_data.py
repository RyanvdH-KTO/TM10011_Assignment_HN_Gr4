# Bestand om data in te laden en data schoon te maken

import pandas as pd
import os
from hn.load_data import load_data

def load_clean_data():
    data = load_data()
    data = pd.DataFrame(data)
    
    return data







