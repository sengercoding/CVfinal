import os
import gdown

def get_gan_checkpoints():
    if not os.path.exists("GAN/checkpoints"):
        print("Creating checkpoints folder")
        os.makedirs("GAN/checkpoints")

    if not os.path.isfile("GAN/checkpoints/res18-unet.pt"):
        print("Downloading ResNet backbone")
        file_id = "11Cakbvp1LFRe6ydryNRzWL-kL067XzrU"
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        gdown.download(url, "GAN/checkpoints/res18-unet.pt", quiet=False)
    else:
        print("ResNet backbone already available")

    if not os.path.isfile("GAN/checkpoints/final_model_weights.pt"):
        print("Downloading base model")
        file_id = "1-5ebrRfvXRaWZAiANDOpaKQMEuBFdP75"
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        gdown.download(url, "GAN/checkpoints/final_model_weights.pt", quiet=False)
    else:
        print("base model already available")

    if not os.path.isfile("GAN/checkpoints/fine_tuned_faces.pt"):
        print("Downloading finetuned model")
        file_id = "1fTzGmFirDF4nofVmvR62NRvqOMZGyEzg"
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        gdown.download(url, "GAN/checkpoints/fine_tuned_faces.pt", quiet=False)
    else:
        print("finetuned model already available")