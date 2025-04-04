from time import time
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
from CNN.src.colorize import test_colorizer_cnn
from GAN.src.colorize import test_colorizer_gan

def visualize(cnn, gan1, gan2, val_path, num_images = 4, seed=44, save = False):
    """
    Visualizes CNN and two GAN models using a path of images to show the ground truth, grayscale, before finetuning, and after finetuning.

    :param model1: the first GAN model used in colorization
    :param model2: the second GAN model used in colorization
    :param val_path: the path of images to visualize colorization on
    :param num_images: the number of images the GAN colorization compares
    :param seed: the set random seed for image colorization
    :param save: saves results to results/face_results_gan_{current time}.jpg if True
    """
    np.random.seed(seed)
    val_paths = glob(val_path + "/*.jpg")
    val_images = np.random.choice(val_paths, num_images, replace = False) # choose 4 random images
    rows = num_images
    cols = 5
    fig, axs = plt.subplots(num_images, 5, figsize=(2*cols,2*rows))
    axs[0,0].set_title("ground truth")
    axs[0,1].set_title("grayscale")
    axs[0,2].set_title("CNN")
    axs[0,3].set_title("GAN before finetuning")
    axs[0,4].set_title("GAN after finetuning")
    for i in range(num_images):
        for j in range(cols):
            axs[i,j].axis("off")
        img = Image.open(val_images[i]).convert("RGB").resize((256,256), Image.BICUBIC)
        axs[i,0].imshow(img)
        colorized_cnn = test_colorizer_cnn(cnn, img)
        grayscale, colorized_gan1 = test_colorizer_gan(gan1, img, return_L=True)
        colorized_gan2 = test_colorizer_gan(gan2, img, return_L=False)
        axs[i,1].imshow(grayscale, cmap="gray")
        axs[i,2].imshow(colorized_cnn)
        axs[i,3].imshow(colorized_gan1)
        axs[i,4].imshow(colorized_gan2)
    fig.tight_layout()
    if save:
        fig.savefig(f"results/colorizations_{time()}.jpg")