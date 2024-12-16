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
def read_lmdb_folder(lmdb_path, folder_name=None):
    """
    Read images from a specific folder key in the LMDB database.

    Parameters:
    - lmdb_path (str): Path to the LMDB database.
    - folder_name (str): The key (folder name) to retrieve images from.
    """
    # print(list_all_keys(lmdb_path))
    if folder_name == None:
        lmdb_file = lmdb_path
    else:
        lmdb_file = os.path.join(lmdb_path, f"{folder_name}.lmdb")
    
    env = lmdb.open(lmdb_file, readonly=True)
    with env.begin() as txn:
        images_data = txn.get("data".encode('ascii'))

    # Deserialize the list of images
    images = pickle.loads(images_data)

    # Convert back from CHW to HWC format for visualization
    # images = [np.transpose(img, (1, 2, 0)) for img in images]

    return images