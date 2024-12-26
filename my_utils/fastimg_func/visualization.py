import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TKAgg')
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 确保减号显示正常
import numpy as np

def display_images(image_list, cols=3, figsize=(15, 10)):
    """
    Display a list of images using matplotlib.

    Args:
        image_list (list of ndarray): List of images to display.
        cols (int): Number of columns in the display grid.
        figsize (tuple): Size of the figure.
    """
    # Calculate number of rows needed
    rows = len(image_list) // cols + int(len(image_list) % cols != 0)

    # Create a figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten in case there are extra subplots

    for img, ax in zip(image_list, axes):
        # Check the shape of the image and convert if necessary
        if len(img.shape) == 2:
            # Grayscale image, no conversion needed
            ax.imshow(img, cmap='gray')
        elif len(img.shape) == 3:
            if img.shape[0] == 1 or img.shape[0] == 3:
                # CHW format, convert to HWC
                img = np.transpose(img, (1, 2, 0))
                ax.imshow(img)
            elif img.shape[2] == 1:
                # HWC with a single channel, display as grayscale
                ax.imshow(img.squeeze(), cmap='gray')
            else:
                # HWC format, no conversion needed
                ax.imshow(img)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        ax.axis('off')  # Hide the axis

    # Hide any remaining subplots
    for ax in axes[len(image_list):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
