import numpy as np
def calculate_stats(errors):
    return {
        'mean': np.mean(errors),
        'median': np.median(errors),
        'std': np.std(errors),
        'max': np.max(errors),
        'min': np.min(errors)
    }

