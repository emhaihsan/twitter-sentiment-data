import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

def convert_params(config):
    for key, value in config.items():
        try:
            config[key] = int(value)
        except ValueError:
            pass
        except TypeError:
            pass
    return config

# Fungsi untuk melakukan stratified sampling pada sparse matrix
def stratified_sample(X, y, sample_size=0.05):
    X = X.copy()
    y = y.copy()
    X_train_sample, _, y_train_sample, _ = train_test_split(X, y, train_size=sample_size, stratify=y, random_state=42)
    # Convert to writable arrays
    return X_train_sample, y_train_sample
