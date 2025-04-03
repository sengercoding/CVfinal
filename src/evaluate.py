import os
import numpy as np
from glob import glob
from PIL import Image
from GAN.src.colorize import test_colorizer_gan
from CNN.src.colorize import test_colorizer_cnn
from tqdm.notebook import tqdm
from skimage.color import rgb2lab, deltaE_ciede2000
from prettytable import PrettyTable


def avg_deltaE(model, val_path, size = 50, model_type = "GAN", seed = 33):
    """
    calculate average deltaE error of a model on validation set

    :param model: tensorflow CNN or torch GAN
    :param val_path: path to validation set
    :param size: number of images to perform validation on
    :param model_type: GAN or CNN
    :param seed: random seed

    :return: average deltaE error

    Note: This is obviously not the most efficient implementation
     (Lab conversion even though output of network is in Lab) but
     provides good ease of use
    """
    np.random.seed(seed)
    val_paths = glob(os.path.join(val_path, "*.jpg"))
    if len(val_paths) < size:
        raise ValueError("size is larger than number of images")
    val_paths = np.random.choice(val_paths, size, replace = False)
    scores = []
    for path in tqdm(val_paths, desc = f"evaluating {model_type}", position=0, leave=True):
        image = Image.open(path).convert("RGB").resize((256,256), Image.BICUBIC)
        if model_type == "GAN":
            colorized = test_colorizer_gan(model, image)
        elif model_type == "CNN":
            colorized = test_colorizer_cnn(model, image)
        image = np.array(image)
        image_lab = rgb2lab(image)
        colorized_lab = rgb2lab(colorized)
        scores.append(deltaE_ciede2000(image_lab, colorized_lab))
    scores = np.array(scores)
    return np.mean(scores)


def mean_chroma(model, val_path, size = 50, model_type = "GAN", seed = 33):
    """
     calculate average mean chroma of an image on a dataset

     :param model: tensorflow CNN or torch GAN
     :param val_path: path to validation set
     :param size: number of images to perform validation on
     :param model_type: GAN or CNN
     :param seed: random seed

     :return: average mean chroma
     """
    np.random.seed(seed)
    val_paths = glob(os.path.join(val_path, "*.jpg"))
    if len(val_paths) < size:
        raise ValueError("size is larger than number of images")
    val_paths = np.random.choice(val_paths, size, replace = False)
    scores = []
    for path in tqdm(val_paths, desc = f"evaluating {model_type}", position=0, leave=True):
        image = Image.open(path).convert("RGB").resize((256,256), Image.BICUBIC)
        if model_type == "GAN":
            colorized = test_colorizer_gan(model, image)
        elif model_type == "CNN":
            colorized = test_colorizer_cnn(model, image)
        image = np.array(image)
        max_vals = np.max(colorized, axis = 2)
        min_vals = np.min(colorized, axis = 2)
        mean_chroma = np.mean(max_vals-min_vals)
        scores.append(mean_chroma)
    scores = np.array(scores)
    return np.mean(scores)


def mse(model, val_path, size = 50, model_type = "GAN", seed = 33):
    """
    calculate average MSE of model on the validation set

    :param model: tensorflow CNN or torch GAN
    :param val_path: path to validation set
    :param size: number of images to perform validation on
    :param model_type: GAN or CNN
    :param seed: random seed

    :return: average MSE
    """
    np.random.seed(seed)
    val_paths = glob(os.path.join(val_path, "*.jpg"))
    if len(val_paths) < size:
        raise ValueError("size is larger than number of images")
    val_paths = np.random.choice(val_paths, size, replace = False)
    scores = []
    for path in tqdm(val_paths, desc = f"evaluating {model_type}", position=0, leave=True):
        image = Image.open(path).convert("RGB").resize((256,256), Image.BICUBIC)
        if model_type == "GAN":
            colorized = test_colorizer_gan(model, image)
        elif model_type == "CNN":
            colorized = test_colorizer_cnn(model, image)
        image = np.array(image)
        error = np.mean((image - colorized)**2)
        scores.append(error)
    scores = np.array(scores)
    return np.mean(scores)

def evaluate_all(cnn, gan1, gan2, val_path, size = 50, seed = 33):
    """
    Evaluate all models for all metrics. Faster than performing evaluation 9 times
    :param cnn: tensorflow CNN
    :param gan1: torch GAN
    :param gan2: torch GAN
    :param val_path: path to validation set
    :param size: number of images to perform validation on
    :param seed: random seed
    """
    results = PrettyTable()
    np.random.seed(seed)
    val_paths = glob(os.path.join(val_path, "*.jpg"))
    if len(val_paths) < size:
        raise ValueError("size is larger than number of images")
    val_paths = np.random.choice(val_paths, size, replace=False)
    deltae_scores_cnn, deltae_scores_gan1, deltae_scores_gan2 = [], [], []
    chroma_scores_cnn, chroma_scores_gan1, chroma_scores_gan2 = [], [], []
    mse_scores_cnn, mse_scores_gan1, mse_scores_gan2 = [], [], []

    for path in tqdm(val_paths, position=0, leave=True):
        image = Image.open(path).convert("RGB").resize((256, 256), Image.BICUBIC)

        # colorize
        cnn_output = test_colorizer_cnn(cnn, image)
        gan1_output = test_colorizer_gan(gan1, image)
        gan2_output = test_colorizer_gan(gan2, image)

        image = np.array(image)

        #convert to Lab
        image_lab = rgb2lab(image/255)
        cnn_output_lab = rgb2lab(cnn_output/255)
        gan1_output_lab = rgb2lab(gan1_output/255)
        gan2_output_lab = rgb2lab(gan2_output/255)

        # calculate delta E
        mean_deltae_cnn = np.mean(deltaE_ciede2000(image_lab, cnn_output_lab))
        mean_deltae_gan1 = np.mean(deltaE_ciede2000(image_lab, gan1_output_lab))
        mean_deltae_gan2 = np.mean(deltaE_ciede2000(image_lab, gan2_output_lab))
        deltae_scores_cnn.append(mean_deltae_cnn)
        deltae_scores_gan1.append(mean_deltae_gan1)
        deltae_scores_gan2.append(mean_deltae_gan2)

        # calculate chromas
        chroma_cnn = np.max(cnn_output, axis = -1) - np.min(cnn_output, axis = -1)
        chroma_gan1 = np.max(gan1_output, axis = -1) - np.min(gan1_output, axis = -1)
        chroma_gan2 = np.max(gan2_output, axis = -1) - np.min(gan2_output, axis = -1)
        mean_chroma_cnn = np.mean(chroma_cnn)
        mean_chroma_gan1 = np.mean(chroma_gan1)
        mean_chroma_gan2 = np.mean(chroma_gan2)
        chroma_scores_cnn.append(mean_chroma_cnn)
        chroma_scores_gan1.append(mean_chroma_gan1)
        chroma_scores_gan2.append(mean_chroma_gan2)

        # calculate MSE
        mse_scores_cnn.append(np.mean((image - cnn_output)**2))
        mse_scores_gan1.append(np.mean((image - gan1_output)**2))
        mse_scores_gan2.append(np.mean((image - gan2_output)**2))

    results.field_names = ["metric", "CNN", "base GAN", "finetuned GAN"]
    results.add_row(["CIEDE 2000 Delta E", round(np.mean(deltae_scores_cnn), 3),
                     round(np.mean(deltae_scores_gan1), 3), round(np.mean(deltae_scores_gan2), 3)])
    results.add_row(["mean average chroma", round(np.mean(chroma_scores_cnn), 3),
                   round(np.mean(chroma_scores_gan1), 3), round(np.mean(chroma_scores_gan2), 3)])
    results.add_row(["MSE", round(np.mean(mse_scores_cnn), 3),
                   round(np.mean(mse_scores_gan1), 3), round(np.mean(mse_scores_gan2), 3)])

    return results