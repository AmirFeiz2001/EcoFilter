import os
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import measure
from datetime import datetime

def plot_confusion_matrix(y_true, y_pred, output_dir="output/plots"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{timestamp}.png"))
    plt.close()
    print(f"Confusion matrix saved to {output_dir}")

def compare_images(imageA, imageB, title, output_dir="output/plots"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mse_val = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)
    ssim_val = measure.compare_ssim(imageA, imageB, multichannel=True)

    plt.figure(figsize=(10, 5))
    plt.suptitle(f"{title} - MSE: {mse_val:.2f}, SSIM: {ssim_val:.2f}")
    plt.subplot(1, 2, 1)
    plt.imshow(imageA, cmap='gray')
    plt.title("Image A")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(imageB, cmap='gray')
    plt.title("Image B")
    plt.axis("off")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(output_dir, f"{title.lower().replace(' ', '_')}_{timestamp}.png"))
    plt.close()
    print(f"Comparison plot saved to {output_dir}")
