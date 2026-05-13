import cv2
import matplotlib.pyplot as plt
import os

def show_image(title, image, cmap=None):
    """Display image in a window"""
    plt.figure(figsize=(10, 6))
    if cmap:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_images_grid(images_dict):
    """Show multiple images in a grid"""
    n = len(images_dict)
    cols = 3
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(18, rows * 5))
    for idx, (title, (img, cmap)) in enumerate(images_dict.items()):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img, cmap=cmap)
        plt.title(title, fontsize=12)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_image(output_path, image):
    """Save result image to disk"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")