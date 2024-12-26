import numpy as np


def iqr_mean(data):
    data = np.array(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    filtered_data = data[(data >= q1 - 1.5 * iqr) & (data <= q3 + 1.5 * iqr)]
    return np.mean(filtered_data)
