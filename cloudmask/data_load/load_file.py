import os
import h5py
import pickle
import numpy as np
import pandas as pd
from cloudmask.utils import _save_npy


class LoadS2Cloud:
    def __init__(self, path_input_data, filename="20160914_s2_manual_classification_data.h5"):
        """
        Module to import and prepare the file "20160914_s2_manual_classification_data.h5"
        :param path_input_data:
        """
        self.path_input_data = path_input_data
        if not os.path.exists(
            self.path_input_data
        ) or filename not in os.listdir(
            self.path_input_data
        ):
            raise ValueError(
                f'The file {filename} is not in the provided path path_input_data'
            )
        self.filename = filename
        self.loaded = False

    def load_file(self):
        """
        Load the .h5 data into a dictionary of arrays
        :return:
        """
        filename = os.path.join(
            self.path_input_data, self.filename
        )
        self.file = h5py.File(filename, "r")

        return {key: np.array(self.file[key]) for key in self.file.keys()}

    def _get_attributes(self):
        """
        Get the relevant attributes from the .h5 file
        :return:
        """
        dict_results = self.load_file()
        self.continent = dict_results["continent"]
        self.country = dict_results["country"]
        self.tiles = dict_results["granule_id"]
        self.date = dict_results["dates"]
        self.longitude = dict_results["longitude"]
        self.latitude = dict_results["latitude"]
        self.spectra = dict_results["spectra"]
        self.classes = dict_results["classes"]
        self.loaded = True

    def get_labels(self):

        if not self.loaded:
            self._get_attributes()

        labels = {
            x: [y.decode("utf8")]
            for x, y in zip(self.file["class_ids"], self.file["class_names"])
        }

        with open(
            os.path.join(self.path_input_data, "labels.pickle"),
            "wb",
        ) as handle:
            pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return labels

    def get_summary_data(self, file_path=None):
        """
        Get a single file at the file level that summarize the informations
        :param save_output:
        :return:
        """
        if not self.loaded:
            self._get_attributes()

        dict_descriptive = pd.DataFrame(
            dict(
                continent=self.continent,
                country=self.country,
                tiles=self.tiles,
                date=self.date,
                longitude=self.longitude,
                latitude=self.latitude,
            )
        )

        location_columns = list(
            dict_descriptive.drop(["longitude", "latitude"], axis=1).columns
        )

        for loc_col in location_columns:
            dict_descriptive[loc_col] = dict_descriptive[loc_col].apply(
                lambda x: x.decode("utf8")
            )

        dict_descriptive["date"] = dict_descriptive["date"].apply(
            lambda x: x.split(" ")[0]
        )
        dict_descriptive["date"].unique()

        dict_location = dict_descriptive.groupby(
            ["continent", "country", "tiles", "date"]
        ).agg("mean")
        dict_location.reset_index(inplace=True)
        if file_path:
            dict_location.to_csv(file_path, index=False)

        return dict_location

    def save_data_per_tile(self, path_output_data="./s2_tiles"):
        """
        Save the data per tile, with X as the spectrum and y as the class
        :param path_output_data:
        :return:
        """

        if not os.path.exists(path_output_data):
            os.mkdir(path_output_data)

        if not self.loaded:
            self._get_attributes()

        # Loop over the tiles and save once all the pixels from a given tile were loaded
        current_tile = None
        dict_arrays = {}

        # Iterate over all data using zip
        for c, co, tile, lo, la, s, cl in zip(
            self.continent,
            self.country,
            self.tiles,
            self.longitude,
            self.latitude,
            self.spectra,
            self.classes,
        ):
            tile_ = tile.decode("utf8")
            dir_path = os.path.join(path_output_data, tile_)

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            if not current_tile:
                current_tile = tile_

            if current_tile != tile_:
                _save_npy(dir_path, dict_arrays[current_tile]["X"], "training_X")
                _save_npy(dir_path, dict_arrays[current_tile]["y"], "training_y")
                _save_npy(dir_path, dict_arrays[current_tile]["lon"], "longitude")
                _save_npy(dir_path, dict_arrays[current_tile]["lat"], "latitude")

                dict_arrays = {}
                current_tile = tile_

            # Add place information to the dictionary
            s = s.reshape(1, s.shape[0])
            cl, lo, la = cl.reshape(1, 1), lo.reshape(1, 1), la.reshape(1, 1)

            # Add spectra to the dictionary
            if tile_ in dict_arrays.keys():
                dict_arrays[tile_]["X"] = np.append(dict_arrays[tile_]["X"], s, axis=0)
                dict_arrays[tile_]["y"] = np.append(dict_arrays[tile_]["y"], cl, axis=0)
                dict_arrays[tile_]["lon"] = np.append(
                    dict_arrays[tile_]["lon"], lo, axis=0
                )
                dict_arrays[tile_]["lat"] = np.append(
                    dict_arrays[tile_]["lat"], la, axis=0
                )
            else:
                dict_arrays[tile_] = {
                    "X": s,
                    "y": cl,
                    "lon": lo,
                    "lat": la,
                }
