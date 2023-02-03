import sys

root_dir = "/home/s999379/git-repo/cloud-mask"
sys.path.insert(0, root_dir)
sys.path.append("../../")  #

import os
from cloudmask.data_load import experiments, load_file

#######################################################################################################

# Load the file 20160914_s2_manual_classification_data.h5 and group pixels by tile name
load_input_file = load_file.LoadS2Cloud(
    path_input_data=os.path.join(root_dir, "examples/data")
)
## Read original file
file = load_input_file.load_file()
file.keys()
## Read labels
load_input_file.get_labels()
## Summarize by location and save it to file_path
dict_location = load_input_file.get_summary_data(
    file_path=os.path.join(root_dir, "data/dict_location.csv")
)

folder_tiles = "s2_tiles"


## Save by tile
load_input_file.save_data_per_tile(
    path_output_data=os.path.join(root_dir, folder_tiles)
)

# Prepare the experiments by splitting train/val/test by groups
exps = experiments.FileExperiments(
    root_dir=root_dir,
    folder_tiles=folder_tiles,
    dict_location=dict_location,
    n_folds=4,  # We seperate data into 5 sets (training:4, testing 1). Then, we split training into 4 validation folds
    test_size=5,
)

## Check if all the tiles have been saved
exps.check_loaded_data()
## Save train/val/test by group
### Group by tile name
exps.save_exp_tiles(folder="experiments_tiles")
### Group by countries
exps.save_exp_countries(folder="experiments_countries")
### Group by continent
exps.save_exp_continents(folder="experiments_continents")
