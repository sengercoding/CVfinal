import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb
from skimage import transform
from PIL import Image
from time import time

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def preprocess_image(image):

    image = np.array(image) / 255.0  # Normalize to [0,1]
    lab = rgb2lab(image).astype("float32")  # Convert to Lab
    
    transform = transforms.ToTensor()
    lab_tensor = transform(lab)
    
    L = lab_tensor[[0], ...] / 50. - 1.  # Scale L to [-1, 1]
    ab = lab_tensor[[1, 2], ...] / 110.  # Scale ab to [-1, 1]

    return L.unsqueeze(0), ab.unsqueeze(0)  # Add batch dimension

def preprocess_bw_image(image):

    image = np.array(image) / 255.0  # Normalize to [0,1]
    lab = rgb2lab(image).astype("float32")  # Convert to Lab
    
    transform = transforms.ToTensor()
    lab_tensor = transform(lab)
    
    L = lab_tensor[[0], ...] / 50. - 1.  # Scale L to [-1, 1]
    ab = lab_tensor[[1, 2], ...] / 110.  # Scale ab to [-1, 1]

    return L.unsqueeze(0), ab.unsqueeze(0)  # Add batch dimension

def test_colorizer(model, image, return_L = False, save=False):
    """
    Turns an image into grayscale and colorizes it
    
    :param model: model that is being used
    :param image: image in PIL.Image format
    :param return_L (optional): returns grayscale version as well, used in visualization
    :param save (optional): saves the image

    :returns L (optional): L* channel (grayscale image)
    :returns colorized: colorized image
    """

    model.net_G.eval()

    image = image.convert("RGB").resize((256,256), Image.BICUBIC)
    
    # Preprocess image: Extract real L and ab channels
    L, ab = preprocess_image(image)
    
    device = next(model.parameters()).device
    L, ab = L.to(device, dtype=torch.float), ab.to(device, dtype=torch.float)

    data = {'L': L, 'ab': ab}

    with torch.no_grad():
        model.setup_input(data)
        model.forward()

    model.net_G.train()
    
    # Get output
    fake_color = model.fake_color.detach()  # Model predicts ab in [-1,1]
    
    # Convert to L*a*b*
    L_real = (L + 1) * 50.0  # Convert L from [-1,1] to [0,100]
    fake_color = fake_color * 110.0  # Convert ab from [-1,1] to [-110,110]

    L_np = L_real[0][0].cpu().numpy()  # Shape: (H, W)
    fake_ab_np = fake_color[0].cpu().numpy().transpose(1, 2, 0)  # Shape: (H, W, 2)

    lab_img = np.zeros((L_np.shape[0], L_np.shape[1], 3))
    lab_img[:, :, 0] = L_np  # Luminance
    lab_img[:, :, 1:] = fake_ab_np  # Predicted color

    # Convert to RGB
    rgb_img = lab2rgb(lab_img)
    rgb_img = (rgb_img * 255).astype(np.uint8)
    
    if save:
        Image.fromarray(rgb_img).save(f"colorized_{time()}")

    if return_L:
        return L_np, rgb_img
    
    else:
        return rgb_img

def test_colorizer_cnn(model, image, return_L = False, save=False):

    img = transform.resize(image, (256, 256), anti_aliasing=True)
    img = np.array(img) / 255.0  # Normalize to [0,1]
    

    # Convert to LAB
    lab = rgb2lab(img)

    # Split into L and AB channels
    L = lab[:,:,0] / 100.0  # Normalize L to [0, 1]
    if L.shape != (256, 256):  # Debugging print
        print(f"Unexpected L shape: {L.shape}")
    L = L.reshape(1, 256, 256, 1)  # Add batch and channel dimensions
    ab_pred = model.predict(L, batch_size=1)
    lab_pred = np.zeros((256, 256, 3))
    lab_pred[:,:,0] = L.squeeze() * 100
    lab_pred[:,:,1:] = ab_pred.squeeze() * 128

    rgb_pred = lab2rgb(lab_pred)
    rgb_pred = transform.resize(rgb_pred, (512, 512), anti_aliasing=True)

    if save:
        Image.fromarray(rgb_pred).save(f"colorized_{time()}")

    return rgb_pred


def colorize(model, image, return_L=False, save=False):
    """
    Colorizes a black-and-white image.

    :param model: Model to use for colorization
    :param image: Grayscale image (PIL.Image)
    :param return_L (optional): Returns grayscale version as well, used in visualization
    :param save (optional): Saves the colorized image

    :returns L (optional): L* channel (grayscale image)
    :returns colorized: Colorized image
    """

    model.net_G.eval()

    image = image.resize((256,256), Image.BICUBIC)

    L, ab = preprocess_bw_image(image)
    
    device = next(model.parameters()).device
    L = L.to(device)

    # Prepare input format
    data = {'L': L, 'ab': ab}

    with torch.no_grad():
        model.setup_input(data)
        model.forward()

    model.net_G.train()
    
    # Get output color channels
    fake_color = model.fake_color.detach()  # Model predicts ab in [-1,1]
    
    # Convert back to Lab scale
    L_real = (L + 1) * 50.0  # Convert L from [-1,1] to [0,100]
    fake_color = fake_color * 110.0  # Convert ab from [-1,1] to [-110,110]

    # Convert to NumPy
    L_np = L_real[0][0].cpu().numpy()  # Shape: (H, W)
    fake_ab_np = fake_color[0].cpu().numpy().transpose(1, 2, 0)  # Shape: (H, W, 2)

    # Construct full Lab image
    lab_img = np.zeros((L_np.shape[0], L_np.shape[1], 3))
    lab_img[:, :, 0] = L_np  # Luminance
    lab_img[:, :, 1:] = fake_ab_np  # Predicted color

    # Convert Lab to RGB
    rgb_img = lab2rgb(lab_img)
    rgb_img = (rgb_img * 255).astype(np.uint8)
    
    if save:
        Image.fromarray(rgb_img).save(f"colorized_{time()}.png")

    if return_L:
        return L_np, rgb_img
    else:
        return rgb_img
