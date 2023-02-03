
<div style="display: flex;">
  <img src=doc/logo.png alt="alt text" title="image title" style="float: right;">
  <p> </p>
</div>


The objective of the *cloud-mask* library is to process and run ML algorithms at the pixel level. It is based on the only publicly available dataset of manually labelled Sentinel-2 scenes for cloud masks.
It gathers 97 tiles with more than 5 million manually labelled pixels, curated by [Hollstein et al](https://www.mdpi.com/2072-4292/8/8/666). The database is provided by **EnMAP** and available on the gitlab repository [Database File of Manually classified Sentinel-2A Data](https://git.gfz-potsdam.de/EnMAP/sentinel2_manual_classification_clouds).

*A report summarizing the findings and results is available [here](/doc/report_s2_manual_classification.pdf).*

# Installation

To install the package, you must first clone the git repository to the desired folder


```bash
git clone git@github.com:j-desloires/s2-pixel-detection.git
```

Then, open Anaconda prompt and create the environment from environment.yml

```
cd cloud-mask
conda env create -f environment.yml
conda activate cloud-mask
pip install .
```

If you want to run the jupyter notebooks, you must set the environment name:

```
ipython kernel install --user --name=cloud-mask
```

# Dependencies

This package has a dependency with 
- [*eo-flow*](https://github.com/sentinel-hub/eo-flow/) python module developped by Sinergise. You must clone and install this library on your environment.
- [*adapt*](https://github.com/adapt-python/adapt) python module developed by Michelin. The method CNN has the class attributes self.encoder, self.task and self.discriminator to be compatible with the library.

# Usage

```python

# Load and prepare experiments
from cloudmask.data_load import load_file, experiments

# Perform the experiments
## Load the prepared train and test dataset
from cloudmask.data_load.utils_exp import load_test_data, load_train_data, subset_data
## Hyperparameters tuning given a standard model with .fit and .predict methods
from cloudmask.base_models.base_ml_tuning import get_predictions, get_dictionary_metrics
## LGBM with focal loss for multiclass classification (one-vs-rest)
from cloudmask.ml_task.ml_models import OneVsRestLightGBMWithCustomizedLoss, FocalLoss
## CNN1D applied on the spectrum dimension
from cloudmask.ml_task import convnet_models

```
You can find out how to call the different modules to run the experiments from python scripts [here](/examples/scripts). It is assumed that you have already downloaded the data from this [link](https://git.gfz-potsdam.de/EnMAP/sentinel2_manual_classification_clouds/-/blob/master/20160914_s2_manual_classification_data.h5) into the [examples/data](/examples/data) folder on the repo.
The scripts are organized to run all the experiments as follows:
1. Data loading (*data_loading.py*)
   1. Load the data and group pixels per tile. All the pixels will be saved into a np.array of (N, 13) dimensions.
   2. Prepare the experiments for ML task to split the data into a training, validation and test sets given a geographical scale (tile, country or continent).
2. Explore the data to compute descriptive statistics (*explore_data.py*).
3. Run standard ml algorithms (e.g. RandomForest) with random_search hyperparameters with leave-one-group-out cross-validation (*ml_tuning.py*).
4. Run LightGBM with Focal loss to take the imbalanced multiclass problem (*lgbm_focal.py*).
5. Run CNN1D models. You can play with the hyperparameters given the dictionary (*deep_learning.py*).
6. Load results for ml algorithms. The metrics were already saved during the training phase (*results_ml.py*). We focus here on LightGBM with focal loss.
7. Load results for deep learning algorithms. We consider the model that gave the best metric on the validation sets for a given test set (must be improved) (*results_deep.py*).


Then you can test the [scripts](/examples/scripts) by changing the *root_path* to where the repo is saved locally.
