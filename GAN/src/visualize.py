from time import time
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
from GAN.src.colorize import test_colorizer

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

def visualize(model1, model2, val_path, num_images = 4, seed=44, save = False):
    np.random.seed(seed)
    val_paths = glob(val_path + "/*.jpg")
    val_images = np.random.choice(val_paths, 4, replace = False) # choose 4 random images
    fig, axs = plt.subplots(num_images,4, figsize=(int(8*num_images/4),8))
    axs[0,0].set_title("ground truth")
    axs[0,1].set_title("grayscale")
    axs[0,2].set_title("before finetuning")
    axs[0,3].set_title("after finetuning")
    for i in range(num_images):
        for j in range(4):
            axs[i,j].axis("off")
        img = Image.open(val_images[i])
        axs[i,0].imshow(img)
        grayscale1, colorized1 = test_colorizer(model1, img, return_L=True)
        colorized2 = test_colorizer(model2, img, return_L=False)
        axs[i,1].imshow(grayscale1, cmap="gray")
        axs[i,2].imshow(colorized1)
        axs[i,3].imshow(colorized2)
    fig.tight_layout()
    if save:
        fig.savefig(f"results/faces_results_gan_{time()}.jpg")
