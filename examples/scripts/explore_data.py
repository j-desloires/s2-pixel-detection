import os
import pandas as pd
from cloudmask.data_load import experiments, load_file
from cloudmask.data_load import experiments, load_file
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#######################################################################################################
root_path = "/home/johann/git-repo/cloud-mask"

# Load the file and save by tile name
load_input_file = load_file.LoadS2Cloud(
    path_input_data=os.path.join(root_path, "examples/data")
)
## Read original file
file = load_input_file.load_file()
file.keys()
## Read labels
labels = load_input_file.get_labels()

# Path where the pixels grouped by tile (np.array) were saved using the script data_loading.py
path_tile = "/home/johann/git-repo/cloud-mask/s2_tiles"
tiles = os.listdir(path_tile)

#######################################################################
# Average spectrum per class

list_means = []
for tile in tiles:

    x = np.load(os.path.join(path_tile, tile + "/training_X.npy"))
    y = np.load(os.path.join(path_tile, tile + "/training_y.npy"))

    means = {}
    for i in np.unique(y):
        tmp = x[np.where(y == i)[0]]
        means[i] = np.mean(tmp, axis=0)

    df_means = pd.DataFrame(means).T
    df_means.columns = file["central_wavelength_nm"]
    df_means["tile"] = tile

    list_means.append(df_means.reset_index())

spectrum_means = pd.concat(list_means, axis=0)
spectrum_means = pd.melt(spectrum_means, id_vars=["tile", "index"])
spectrum_means = spectrum_means[spectrum_means["variable"] != 1380]

labels = pd.DataFrame(labels).T.reset_index()
labels.columns = ["index", "class"]
spectrum_means = pd.merge(spectrum_means, labels, on="index", how="left")

###########################

sns.lineplot(
    data=spectrum_means.drop(["tile", "index"], axis=1),
    x="variable",
    y="value",
    hue="class",
)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.show()

################################################################
# Analyze where tiles are distributed

dict_location = load_input_file.get_summary_data(
    file_path=os.path.join(root_path, "data/dict_location.csv")
)

count_tiles = (
    dict_location[["continent", "tiles"]]
    .groupby("continent")
    .agg("count")
    .reset_index()
)

sns.barplot(x="continent", y="tiles", hue="continent", data=count_tiles)
plt.legend(loc="upper center")
plt.show()
count_countries = (
    dict_location[["continent", "country"]]
    .drop_duplicates()
    .groupby("continent")
    .agg("count")
)
sns.barplot(x="continent", y="tiles", hue="continent", data=count_tiles)
plt.legend(loc="upper center")
plt.show()

count = pd.merge(count_countries, count_tiles, on="continent", how="left")
count.sum(axis=0)

################################
# Number of pixels per class and per continent
dict_continent = {}
for tile, continent in zip(dict_location["tiles"], dict_location["continent"]):

    if os.path.exists(os.path.join(path_tile, tile + "/training_y.npy")):
        y = np.load(os.path.join(path_tile, tile + "/training_y.npy"))

        means = {}
        for i in np.unique(y):
            tmp = y[np.where(y == i)[0]]
            means[i] = tmp.shape[0]

        if continent in dict_continent:
            for key, value in means.items():
                if key in dict_continent[continent][0].keys():
                    dict_continent[continent][0][key] += value
                else:
                    dict_continent[continent][0][key] = value
        else:
            dict_continent[continent] = [means]

ls_df = [pd.DataFrame(value_).T for value_ in dict_continent.values()]
ls_df = pd.concat(ls_df, axis=1)
ls_df.columns = dict_continent.keys()
ls_df.index = ["Clear", "Water", "Snow", "Cirrus", "Cloud", "Shadow"]
share_pixels = ls_df / ls_df.sum(axis=0)
share_pixels = round(share_pixels, 3)
share_pixels.sum(axis=1)
print(share_pixels.to_latex(index=True))
