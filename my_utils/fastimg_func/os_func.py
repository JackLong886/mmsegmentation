import psutil


def get_available_men_in_mb():
    men = psutil.virtual_memory().available
    return men // 1024 // 1024


def get_available_men_in_gb():
    men = psutil.virtual_memory().available
    return men // 1024 // 1024 // 1024


if __name__ == '__main__':
    men = get_available_men_in_mb()
    print(men)
