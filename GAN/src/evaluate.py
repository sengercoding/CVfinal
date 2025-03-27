import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
from GAN.src.colorize import colorize_image
from tqdm import tqdm
from skimage.color import rgb2lab, deltaE_ciede2000

def plot_finetuning_loss(save = True):
    # load losses exported from google colab
    loss_G = np.load("results/loss_G.npy")
    loss_G_L1 = np.load("results/loss_G_L1.npy")
    loss_G_GAN = np.load("results/loss_G_GAN.npy")
    epoch_starts = np.arange(1,3750,375)
    epoch_range = np.arange(1,epoch_starts.size+1)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (9,3))
    ax1.plot(epoch_range, loss_G_GAN[epoch_starts], label = r"$\mathcal{L}_{cGAN}$", color = "blue")
    ax2.plot(epoch_range, loss_G_L1[epoch_starts], label = r"$\mathcal{L}_{L_1}$", color = "green")
    ax2.plot(epoch_range, loss_G[epoch_starts], label = r"$\mathcal{L}$", color = "orange")
    ax1.set_xlabel("epoch")
    ax2.set_xlabel("epoch")
    fig.tight_layout()
    ax1.legend()
    ax2.legend()
    if save:
        fig.savefig("results/finetuning_loss.jpg", dpi = 400)
    plt.show()

def visualize(model1, model2, seed=44):
    np.random.seed(seed)
    val_paths = glob("GAN/data/faces_val/*.jpg")
    val_images = np.random.choice(val_paths, 4, replace = False) # choose 4 random images
    fig, axs = plt.subplots(4,4, figsize=(8,8))
    axs[0,0].set_title("ground truth")
    axs[0,1].set_title("grayscale")
    axs[0,2].set_title("before finetuning")
    axs[0,3].set_title("after finetuning")
    for i in range(4):
        for j in range(4):
            axs[i,j].axis("off")
        img = Image.open(val_images[i])
        axs[i,0].imshow(img)
        grayscale1, colorized1 = colorize_image(model1, img, return_L=True)
        colorized2 = colorize_image(model2, img, return_L=False)
        axs[i,1].imshow(grayscale1, cmap="gray")
        axs[i,2].imshow(colorized1)
        axs[i,3].imshow(colorized2)
    fig.savefig("results/faces_results_gan.jpg")


def avg_deltaE(model, val_path, size = 50, seed = 33):
    np.random.seed(seed)
    val_paths = glob(os.path.join(val_path, "*.jpg"))
    if len(val_paths) < size:
        raise ValueError("size is larger than number of images")
    val_paths = np.random.choice(val_paths, size, replace = False)
    scores = []
    for path in tqdm(val_paths):
        image = Image.open(path)
        image = np.array(image)
        image_lab = rgb2lab(image)
        colorized = colorize_image(model, image)
        colorized_lab = rgb2lab(colorized)
        scores.append(deltaE_ciede2000(image_lab, colorized_lab))
    scores = np.array(scores)
    return np.mean(scores)

def mean_chroma(model, val_path, size = 50, seed = 33):
    np.random.seed(seed)
    val_paths = glob(os.path.join(val_path, "*.jpg"))
    if len(val_paths) < size:
        raise ValueError("size is larger than number of images")
    val_paths = np.random.choice(val_paths, size, replace = False)
    scores = []
    for path in tqdm(val_paths):
        image = Image.open(path)
        image = np.array(image)
        colorized = colorize_image(model, image)
        max_val = np.max(colorized, axis = 2)
        min_val = np.min(colorized, axis = 2)
        mean_chroma = np.mean(max_val-min_val)
        scores.append(mean_chroma)
    scores = np.array(scores)
    return np.mean(scores)