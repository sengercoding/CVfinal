import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb
from PIL import Image

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

def colorize_image(model, image, return_L = False, save=False):
    """
    Colorize a single color image using the model and compare with the original.
    
    Args:
        model: Trained colorization model.
        image: PIL image (color).
        save: Whether to save the output image.
    """

    model.net_G.eval()
    
    # Preprocess image: Extract real L and ab channels
    L, ab = preprocess_image(image)
    
    # Move to the correct device
    device = next(model.parameters()).device
    L, ab = L.to(device, dtype=torch.float), ab.to(device, dtype=torch.float)

    # Prepare input format
    data = {'L': L, 'ab': ab}

    with torch.no_grad():
        model.setup_input(data)
        model.forward()

    model.net_G.train()
    
    # Get output
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
    
    if save:
        rgb_img

    if return_L:
        return L_np, rgb_img
    else:
        return rgb_img
    


import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb
from PIL import Image

def preprocess_image(image):

    image = np.array(image) / 255.0  # Normalize to [0,1]
    lab = rgb2lab(image).astype("float32")  # Convert to Lab
    
    transform = transforms.ToTensor()
    lab_tensor = transform(lab)
    
    L = lab_tensor[[0], ...] / 50. - 1.  # Scale L to [-1, 1]
    ab = lab_tensor[[1, 2], ...] / 110.  # Scale ab to [-1, 1]

    return L.unsqueeze(0), ab.unsqueeze(0)  # Add batch dimension

def visualize_single(model, image, save=False):

    model.net_G.eval()
    
    # Preprocess image: Extract real L and ab channels
    L, ab = preprocess_image(image)
    
    # Move to the correct device
    device = next(model.parameters()).device
    L, ab = L.to(device, dtype=torch.float), ab.to(device, dtype=torch.float)

    # Prepare input format
    data = {'L': L, 'ab': ab}

    with torch.no_grad():
        model.setup_input(data)
        model.forward()

    model.net_G.train()
    
    # Get output
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

    # Convert original image to NumPy for visualization
    original_img = np.array(image) / 255.0  

    # Plot results
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(L_np, cmap='gray')
    ax[0].set_title("Input Luminance (Grayscale)")
    ax[0].axis("off")

    ax[1].imshow(rgb_img)
    ax[1].set_title("Colorized Output")
    ax[1].axis("off")

    ax[2].imshow(original_img)
    ax[2].set_title("Original Image")
    ax[2].axis("off")

    plt.show()
    
    if save:
        fig.savefig(f"colorized_{time.time()}.png")

    return rgb_img  # Return colorized image as NumPy array