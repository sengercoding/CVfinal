from PIL import Image
import os
import warnings

def png2jpg(in_dir, out_dir):
    """
    converts a directory of png images to jpg
    """
    os.makedirs(out_dir, exist_ok=True)
    for img_path in os.listdir(in_dir):
        if img_path.endswith(".png"):
            # convert pngs, just copy jpgs
            img_path_long = os.path.join(in_dir, img_path)
            img = Image.open(img_path_long)
            rgb_img = img.convert('RGB')
            out_path = os.path.join(out_dir, img_path.removesuffix(".png") + ".jpg")
            rgb_img.save(out_path)

        elif img_path.endswith(".jpg"):
            img_path_long = os.path.join(in_dir, img_path)
            img = Image.open(img_path_long)
            rgb_img = img.convert('RGB')
            out_path = os.path.join(out_dir, img_path)
            rgb_img.save(out_path)

        else:
            warnings.warn(f"{os.path.join(in_dir, img_path)} is not a png or jpg")
            continue

