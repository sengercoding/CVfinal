from time import time
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
from GAN.src.colorize import test_colorizer, test_colorizer_cnn

def plot_finetuning_loss(save = True):
    """
    Loads losses exported from Google Colab and plots them 

    :param save=True: saves results to results/finetuning_loss.jpg if True

    :returns None, but uses matplotlib to plot a graph
    
    """

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

def visualize_gan(model1, model2, val_path, num_images = 4, seed=44, save = False):
    """
    Visualizes two GAN models using a path of images to show the ground truth, grayscale, before finetuning, and after finetuning
    for four images.

    :param model1: the first GAN model used in colorization
    :param model2: the second GAN model used in colorization
    :param val_path: the path of images to visualize colorization on
    :param num_images: the number of images the GAN colorization compares
    :param seed: the set random seed for image colorization
    :param save: saves results to results/face_results_gan_{current time}.jpg if True

    :returns None, but uses matplotlib to plot a graph
    
    """
    np.random.seed(seed)
    val_paths = glob(val_path + "/*.jpg")
    val_images = np.random.choice(val_paths, num_images, replace = False) # choose 4 random images
    rows = num_images
    cols = 4
    fig, axs = plt.subplots(num_images,4, figsize=(2*cols, 2*rows))
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