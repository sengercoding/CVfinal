from time import time
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
from CNN.src.colorize import test_colorizer_cnn

def visualize_cnn(model, val_path, num_images = 4, seed = 44, save = False):
    """
    Visualizes one CNN model using a path of images to show the ground truth, grayscale, and colorization for four images.

    :param model: the CNN model used in colorization
    :param val_path: the path of images to visualize colorization on
    :param num_images: the number of images the CNN colorization compares
    :param seed: the set random seed for image colorization
    :param save: saves results to results/face_results_cnn_{current time}.jpg if True

    :returns None, but uses matplotlib to plot a graph
    
    """
    np.random.seed(seed)
    val_paths = glob(val_path + "/*.jpg")
    val_images = np.random.choice(val_paths, num_images, replace = False) # choose 4 random images
    rows = num_images
    cols = 3
    fig, axs = plt.subplots(num_images,4, figsize=(2*cols, 2*rows))
    axs[0,0].set_title("ground truth")
    axs[0,1].set_title("grayscale")
    axs[0,2].set_title("cnn colorization")

    for i in range(num_images):
        for j in range(4):
            axs[i,j].axis("off")
        img = Image.open(val_images[i])
        axs[i,0].imshow(img)
        resized_img = img.convert("RGB").resize((256,256), Image.BICUBIC)
        grayscale1, colorized1 = test_colorizer_cnn(model, resized_img, return_L = True)
        axs[i,1].imshow(grayscale1, cmap="gray") # Does a grayscale version need to be created if the colorized is just grayscale with color?
        axs[i,2].imshow(colorized1)
    fig.tight_layout()

    if save:
        fig.savefig(f"results/faces_results_cnn_{time()}.jpg")