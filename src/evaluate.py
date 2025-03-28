import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
from GAN.src.colorize import test_colorizer, test_colorizer_cnn
from tqdm import tqdm
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import mean_squared_error


def avg_deltaE(model, val_path, size = 50, model_type = "GAN", seed = 33):
    np.random.seed(seed)
    val_paths = glob(os.path.join(val_path, "*.jpg"))
    if len(val_paths) < size:
        raise ValueError("size is larger than number of images")
    val_paths = np.random.choice(val_paths, size, replace = False)
    scores = []
    for path in tqdm(val_paths):
        image = Image.open(path).convert("RGB").resize((256,256), Image.BICUBIC)
        if model_type == "GAN":
            colorized = test_colorizer(model, image)
        elif model_type == "CNN":
            colorized = test_colorizer_cnn(model, image)
        image = np.array(image)
        image_lab = rgb2lab(image)
        colorized_lab = rgb2lab(colorized)
        scores.append(deltaE_ciede2000(image_lab, colorized_lab))
    scores = np.array(scores)
    return np.mean(scores)


def mean_chroma(model, val_path, size = 50, model_type = "GAN", seed = 33):
    np.random.seed(seed)
    val_paths = glob(os.path.join(val_path, "*.jpg"))
    if len(val_paths) < size:
        raise ValueError("size is larger than number of images")
    val_paths = np.random.choice(val_paths, size, replace = False)
    scores = []
    for path in tqdm(val_paths):
        image = Image.open(path).convert("RGB").resize((256,256), Image.BICUBIC)
        if model_type == "GAN":
            colorized = test_colorizer(model, image)
        elif model_type == "CNN":
            colorized = test_colorizer_cnn(model, image)
        image = np.array(image)
        max_val = np.max(colorized, axis = 2)
        min_val = np.min(colorized, axis = 2)
        mean_chroma = np.mean(max_val-min_val)
        scores.append(mean_chroma)
    scores = np.array(scores)
    return np.mean(scores)

def mse(model, val_path, size = 50, model_type = "GAN", seed = 33):
    np.random.seed(seed)
    val_paths = glob(os.path.join(val_path, "*.jpg"))
    if len(val_paths) < size:
        raise ValueError("size is larger than number of images")
    val_paths = np.random.choice(val_paths, size, replace = False)
    scores = []
    for path in tqdm(val_paths):
        image = Image.open(path).convert("RGB").resize((256,256), Image.BICUBIC)
        if model_type == "GAN":
            colorized = test_colorizer(model, image)
        elif model_type == "CNN":
            colorized = test_colorizer_cnn(model, image)
        image = np.array(image)
        scores.append(mean_squared_error(image,colorized))
    scores = np.array(scores)
    return np.mean(scores)