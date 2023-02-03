import os
import random
import warnings

import numpy as np
from cloudmask.utils import _save_npy


########################################################################################


class FileExperiments:
    def __init__(
        self, root_dir, dict_location, folder_tiles="s2_tiles", test_size=5, n_folds=4
    ):
        """
        Prepare experiments train/val/test according to a type of cross-validation (by tile, country or continent)
        :param root_dir (str) : path where numpy tiles are saved
        :param dict_location (pd.DataFrame) : dataframe with metadata regarding the tile and their location
        :param test_size (int) : number of disjoint test sets (e.g. 5 means that one test_set is ~20% of the observation)
        :param n_folds (int) : number of disjoint validation folds (e.g. 4 means a validation fold is ~25% of training obs). Only application if not continents
        """
        self.root_dir = root_dir
        self.folder = folder_tiles
        self.folder_path = os.path.join(root_dir, folder_tiles)
        self.dict_location = dict_location
        self.test_size = test_size
        self.n_folds = n_folds
        self.filtered = False

    def check_loaded_data(self):
        """
        Check if the file have been loaded and if the all data is disaggregated per tile
        :return:
        """

        tiles = [k for k in os.listdir(self.folder_path) if "T" in k]
        valid_tiles = [
            tile
            for tile in tiles
            if len(os.listdir(os.path.join(self.folder_path, tile))) > 0
        ]
        self.dict_location["valid_folder"] = [
            tile_ in valid_tiles for tile_ in self.dict_location["tiles"]
        ]
        n_tiles = self.dict_location.shape[0]
        n_valid_tiles = self.dict_location[self.dict_location["valid_folder"]].shape[0]

        if n_tiles > n_valid_tiles > 0:
            warnings.warn(
                f'Among the {str(self.dict_location.shape[0])}, only {str(self.dict_location[self.dict_location["valid_folder"]].shape[0])} tile are valid folders'
            )
        elif n_valid_tiles == 0:
            raise ValueError("Any of the tile has been yet saved")

        self.dict_location = self.dict_location[self.dict_location["valid_folder"]]
        self.filtered = True

    def _check_data(self):
        """
        Raise warning message if the load_file.py module produced the files expected
        :return:
        """
        if not self.filtered:
            raise ValueError(
                "You must first check the location DataFrame with the folders where the tiles are saved using check_loaded_data() method"
            )

    def _load_data(self, tile, file="training_X"):
        """
        Load spectrum or the class of pixels sampled from a given tile
        :param tile: name of the tile
        :param file: type of file (training_X for spectrum or training_y for the class )
        :return:
        """
        if file not in ["training_X", "training_y"]:
            raise ValueError("The file must be either 'training_X' or 'training_y'")
        path_file = os.path.join(self.folder_path, os.path.join(tile, f"{file}.npy"))
        return np.load(path_file)

    def _load_tiles(self, tiles, file="training_X"):
        """
        Concatenate all the file from a given list of tiles
        :param tiles: list of tiles
        :param file: type of the file (training_X for spectrum or training_y for the class )
        :return:
        """
        x = [self._load_data(tile, file) for tile in tiles]
        groups = [[tile] * x_.shape[0] for x_, tile in zip(x, tiles)]
        x = np.concatenate(x, axis=0)
        groups = np.concatenate(groups, axis=0)
        return x, groups

    def _load_and_save_xy(self, saving_path, subset_tiles, category="train"):
        """
        Load spectrum and the class from a given list of tiles and concatenate them to form a set (training or val or test)
        :param saving_path: path where tiles are saved
        :param subset_tiles: list of tiles to aggregate
        :param category: type of set from the concatenated files. It will be the name of the saved .npy file
        :return:
        """

        training_x, groups = self._load_tiles(subset_tiles, file="training_X")
        _save_npy(saving_path, training_x, f"x_{category}")
        training_y, _ = self._load_tiles(subset_tiles, file="training_y")
        _save_npy(saving_path, training_y, f"y_{category}")
        _save_npy(saving_path, np.array(groups), f"meta_{category}")

    def save_exp_tiles(self, folder="experiments_tiles"):
        """
        Save experiments using train/val/test from randomly sampled tiles
        :param folder: folder name for the output of this experiment
        :return:
        """
        self._check_data()
        # Separate randomly tile
        output_path = os.path.join(self.root_dir, folder)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        tiles = self.dict_location["tiles"].unique()
        random.Random(0).shuffle(tiles)

        # We define 5 test set => number of tiles per test
        split_test = len(tiles) // self.test_size

        # From the 4 remaining, we define 10 folds
        for i in range(self.test_size):

            test_path = os.path.join(output_path, f"test_{str(i + 1)}")

            if not os.path.exists(test_path):
                os.mkdir(test_path)

            test_tiles = tiles[(split_test * i) : (split_test * (i + 1))]
            self._load_and_save_xy(
                subset_tiles=test_tiles, saving_path=test_path, category="test"
            )
            train_tiles = [tile_ for tile_ in tiles if tile_ not in test_tiles]

            split_val = len(train_tiles) // self.n_folds
            for j in range(self.n_folds):

                path_fold = os.path.join(test_path, f"fold_{str(j + 1)}")
                if not os.path.exists(path_fold):
                    os.mkdir(path_fold)

                fold = train_tiles[(split_val * j) : (split_val * (j + 1))]
                filter_idx = np.isin(train_tiles, fold)
                training_tiles, val_tiles = (
                    np.array(train_tiles)[~filter_idx],
                    np.array(train_tiles)[filter_idx],
                )
                self._load_and_save_xy(
                    saving_path=path_fold, subset_tiles=training_tiles, category="train"
                )
                self._load_and_save_xy(
                    saving_path=path_fold, subset_tiles=val_tiles, category="val"
                )

    def save_exp_countries(self, folder="experiments_countries"):
        """
        Save experiments using train/val/test from randomly sampled countries
        :param folder: folder name for the output of this experiment
        :return:
        """
        self._check_data()
        output_path = os.path.join(self.root_dir, folder)

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        countries = list(self.dict_location["country"].unique())
        # We define 5 test set
        split_test = len(countries) // self.test_size

        for i in range(self.test_size):
            test_path = os.path.join(output_path, f"test_{str(i + 1)}")
            if not os.path.exists(test_path):
                os.mkdir(test_path)
            # tiles for the test set
            test_countries = countries[(split_test * i) : (split_test * (i + 1))]
            test_tiles = self.dict_location.loc[
                self.dict_location.country.isin(test_countries), "tiles"
            ]

            self._load_and_save_xy(
                saving_path=test_path, subset_tiles=test_tiles, category="test"
            )
            train_countries = self.dict_location.loc[
                ~self.dict_location.country.isin(test_countries), "country"
            ].unique()
            split_val = len(train_countries) // self.n_folds

            for j in range(self.n_folds):
                path_fold = os.path.join(test_path, f"fold_{str(j + 1)}")
                if not os.path.exists(path_fold):
                    os.mkdir(path_fold)

                fold = train_countries[(split_val * j) : (split_val * (j + 1))]

                training_tiles = [
                    tile_
                    for tile_, country in zip(
                        self.dict_location["tiles"], self.dict_location["country"]
                    )
                    if country not in fold and country in train_countries
                ]

                val_tiles = [
                    tile_
                    for tile_, country in zip(
                        self.dict_location["tiles"], self.dict_location["country"]
                    )
                    if country in fold
                ]

                self._load_and_save_xy(
                    saving_path=path_fold, subset_tiles=training_tiles, category="train"
                )
                self._load_and_save_xy(
                    saving_path=path_fold, subset_tiles=val_tiles, category="val"
                )

    def save_exp_continents(self, folder="experiments_continent"):
        """
        Save experiments using train/val/test from randomly sampled continents
        :param folder: folder name for the output of this experiment
        :return:
        """

        # Separate by continent
        self._check_data()
        output_path = os.path.join(self.root_dir, folder)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        continents = self.dict_location["continent"].unique()

        for i, c in enumerate(continents):
            test_path = os.path.join(output_path, f"test_{str(i + 1)}")
            if not os.path.exists(test_path):
                os.mkdir(test_path)

            test_tiles = self.dict_location.loc[
                self.dict_location.continent == c, "tiles"
            ]
            self._load_and_save_xy(
                saving_path=test_path, subset_tiles=test_tiles, category="test"
            )

            train_continents = [k for k in continents if k != c]

            for j, val_c in enumerate(train_continents):

                path_fold = os.path.join(test_path, f"fold_{str(j + 1)}")
                if not os.path.exists(path_fold):
                    os.mkdir(path_fold)

                training_tiles = self.dict_location.loc[
                    (self.dict_location.continent.isin(train_continents))
                    & (self.dict_location.continent != val_c),
                    "tiles",
                ]
                val_tiles = self.dict_location.loc[
                    self.dict_location.continent == val_c, "tiles"
                ]

                self._load_and_save_xy(
                    saving_path=path_fold,
                    subset_tiles=training_tiles,
                    category="train",
                )
                self._load_and_save_xy(
                    saving_path=path_fold, subset_tiles=val_tiles, category="val"
                )
