from PIL import Image
from PIL import ImageEnhance
import PIL
import random
import numpy as np
import gzip
import pickle
import lmdb
import os
import cv2

def data_augmentation(resize=(320, 240), crop_size=224, is_train=True):
    if is_train:
        left, top = np.random.randint(0, resize[0] - crop_size), np.random.randint(0, resize[1] - crop_size)
    else:
        left, top = (resize[0] - crop_size) // 2, (resize[1] - crop_size) // 2

    return (left, top, left + crop_size, top + crop_size), resize


class Brightness(object):
    def __init__(self, min=1, max=1) -> None:
        self.min = min
        self.max = max
    def __call__(self, clip):
        factor = random.uniform(self.min, self.max)
        if isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        new_clip = []
        for img in clip: 
            enh_bri = ImageEnhance.Brightness(img)
            new_img = enh_bri.enhance(factor=factor)
            new_clip.append(new_img)
        return new_clip

class Color(object):
    def __init__(self, min=1, max=1) -> None:
        self.min = min
        self.max = max
    def __call__(self, clip):
        factor = random.uniform(self.min, self.max)
        if isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        new_clip = []
        for img in clip: 
            enh_col = ImageEnhance.Color(img)
            new_img = enh_col.enhance(factor=factor)
            new_clip.append(new_img)
        return new_clip
    
def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

def list_all_keys(lmdb_path, folder_name=None):
    """
    Lists all keys in the specified LMDB database (or a specific sub-LMDB).

    Parameters:
    - lmdb_path (str): Path to the LMDB database.
    - folder_name (str): Optional folder name (sub-LMDB).
    """
    if folder_name is None:
        lmdb_file = lmdb_path
    else:
        lmdb_file = os.path.join(lmdb_path, f"{folder_name}.lmdb")

    try:
        env = lmdb.open(lmdb_file, readonly=True, lock=False)
        with env.begin() as txn:
            keys = [key.decode('ascii') for key, _ in txn.cursor()] # Decode bytes to string
        env.close()
        return keys
    except lmdb.Error as e:
        print(f"Error accessing LMDB file {lmdb_file}: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occured: {e}")
        return []

def read_lmdb_folder(lmdb_dataset_path, folder_name=None):
    """
    Read images from a specific folder key in the LMDB database.

    Parameters:
    - lmdb_path (str): Path to the LMDB database.
    - folder_name (str): The key (folder name) to retrieve images from.
    """
    # print(list_all_keys(lmdb_path))
    
    lmdb_path = os.path.join(lmdb_dataset_path, f"{folder_name}.lmdb")
    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            # value = []
            for key, value in cursor:
                return np.frombuffer(value, dtype=np.float32)  # Return the list of NumPy tensors

    except Exception as e:
        print(f"Error reading LMDB database {lmdb_path}: {e}")
        return None

def csldaily_read_lmdb_folder(lmdb_dataset_path, folder_name):
    """
    Read images from a specific folder LMDB database and return them as a PyTorch tensor.
    Also prints the keys (image filenames).

    Parameters:
    - lmdb_path (str): Path to the LMDB database.
    - folder_name (str): The key (folder name) to retrieve images from.

    Returns:
    - A torch tensor of shape (N, C, H, W)
    - A list of keys (filenames)
    """
    lmdb_path = os.path.join(lmdb_dataset_path, folder_name)
    # print(lmdb_path)
    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            frames = []
            for key, value in cursor:
                img_np = np.frombuffer(value, dtype=np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)  # Ensure RGB decode
                if img is None:
                    print(f"Warning: Could not decode image with key {key}")
                    return None
                frames.append(img)
            return frames  # Return the list of NumPy tensors

    except Exception as e:
        print(f"Error reading LMDB database {lmdb_path}: {e}")
        return None