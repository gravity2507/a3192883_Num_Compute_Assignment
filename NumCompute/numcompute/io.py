
import numpy as np

def read_csv(path, delimiter=",", skip_header=1, missing=np.nan):
    
    data = np.genfromtxt(path, delimiter=delimiter, skip_header=skip_header,
                         filling_values=missing, ndmin=2)
    return data

def iter_csv_chunks(path, chunk_size=50, **kwargs):
    
    if chunk_size < 1: raise ValueError("chunk_size must be positive")
    data = read_csv(path, **kwargs)
    for start in range(0, len(data), chunk_size):
        yield data[start:start + chunk_size]
