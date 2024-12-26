import numpy as np


def get_1d_exts(start, end, step):
    exts = np.arange(start, end, step)
    if exts[-1] != end - 1:
        exts = np.append(exts, end-1)
    return exts


if __name__ == '__main__':
    print(get_1d_exts(0, 20, 7))
