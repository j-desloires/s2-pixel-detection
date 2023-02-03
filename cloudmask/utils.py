import os
import array
import shutil
import numpy as np


def _save_npy(path, array, name_array):
    with open(os.path.join(path, f"{name_array}.npy"), "wb") as f:
        np.save(f, array)


def _remove_empty_folder(root_dir):
    tiles = os.listdir(root_dir)
    tiles = [k for k in tiles if "T" in k]
    for tile in tiles:
        path_file = os.path.join(root_dir, tile)
        dir = os.listdir(path_file)
        # Checking if the list is empty or not
        if len(dir) == 0:
            print("Empty directory")
            shutil.rmtree(path_file)
