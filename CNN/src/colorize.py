import numpy as np
from skimage.color import rgb2lab, lab2rgb
from PIL import Image
from time import time

def test_colorizer_cnn(model, image, return_L = False, save=False):
    """
    Turns an image into grayscale and colorizes it using the CNN
    A seperate function is needed because of differing model makeup
    
    :param model: model that is being used
    :param image: image in PIL.Image format
    :param return_L (optional): returns grayscale version as well, used in visualization
    :param save (optional): saves the image

    :returns L (optional): L* channel (grayscale image)
    :returns colorized: colorized image
    """

    #img = transform.resize(image, (256, 256), anti_aliasing=True)
    img = np.array(image) / 255.0  # Normalize to [0,1]
    
    # Convert to LAB
    lab = rgb2lab(img)

    # Split into L and AB channels
    L = lab[:,:,0] / 100.0  # Normalize L to [0, 1]
    L_np = L
    if L.shape != (256, 256):  # Debugging print
        print(f"Unexpected L shape: {L.shape}")
    L = L.reshape(1, 256, 256, 1)  # Add batch and channel dimensions
    ab_pred = model.predict(L, batch_size=1, verbose=0)
    lab_pred = np.zeros((256, 256, 3))
    lab_pred[:,:,0] = L.squeeze() * 100
    lab_pred[:,:,1:] = ab_pred.squeeze() * 128

    rgb_pred = lab2rgb(lab_pred)
    rgb_pred = (rgb_pred * 255).astype(np.uint8)

    if save:
        Image.fromarray(rgb_pred).save(f"colorized_{time()}")

    if return_L:
        return L_np, rgb_pred
    
    else:
        return rgb_pred