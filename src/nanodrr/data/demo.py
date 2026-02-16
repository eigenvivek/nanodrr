import os
import torch
from platformdirs import user_cache_dir

CACHE_DIR = user_cache_dir("nanodrr")


def download_deepfluoro(subject: int = 1) -> tuple[str, str]:
    """Download a subject from the DeepFluoro dataset."""
    subject = f"subject{subject:02d}"
    base_url = f"https://huggingface.co/datasets/eigenvivek/xvr-data/resolve/main/deepfluoro/{subject}"
    imagepath = os.path.join(CACHE_DIR, "deepfluoro", subject, "volume.nii.gz")
    labelpath = os.path.join(CACHE_DIR, "deepfluoro", subject, "mask.nii.gz")

    for url, local_path in [
        (f"{base_url}/volume.nii.gz", imagepath),
        (f"{base_url}/mask.nii.gz", labelpath),
    ]:
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            torch.hub.download_url_to_file(url, local_path)

    return imagepath, labelpath
