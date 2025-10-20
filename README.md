<img src="img/city_emblem.png" alt="City Logo"/>

# City of Cape Town - Data Science Unit Code Challenge

Solution by Stefan Strydom (stefan.strydom87@gmail.com)

## Shortlisted positions
- Head: Data Science
- Senior Professional Officer: Data Science

## Setup

### Install `uv`
`uv` is the recommended package manager. To install `uv` on Linux or MacOS, run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Or to use `wget` to download the script, use:
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```
To install `uv` on Windows, execute the following in `PowerShell`:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For detailed installation instructions, see the [uv docs](https://docs.astral.sh/uv/getting-started/installation/).

### Clone this repository
```bash
git clone https://github.com/stefan027/ds_code_challenge.git
```

### Install dependencies
Use `uv` to create a virtual environment and to install all dependencies in the virtual environment. The commands below creates the environment `.venv` in the current directory.

```bash
cd ds_code_challenge
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Data requirements
All data are assumed to be in the `./data` directory. A script to create the `./data` directory and to download the data is provided. To run the script, AWS credentials must already be configured. Alternatively, the dummy credentials file can be downloaded with `wget`:
```bash
wget -O .ds_code_challenge_creds.json https://cct-ds-code-challenge-input-data.s3.af-south-1.amazonaws.com/ds_code_challenge_creds.json
```

To download all the data from S3 using the dummy credentials file, run the following:
```bash
python scripts/download_data.py -c .ds_code_challenge_creds.json
```

If AWS credentials were already configured, simply run:
```bash
python scripts/download_data.py
```

## Initial data transformation task
Both a Python script and Jupyter notebook are available for this task. The Python script can be found in [`./scripts/data_transformation.py`](./scripts/data_transformation.py). The notebook can be found in [`./notebooks/data_transformation.ipynb`](./notebooks/data_transformation.ipynb).

When running the notebook, the paths to the relevant input files as well as the error threshold can be set by changing the constants at the top of the notebook.

The Python script can be run as follows:
```bash
python scripts/data_transformation.py -d <path-to-data-directory> -e <error-threshold>
```
For example:
```bash
python scripts/data_transformation.py -d ./data -e 0.05
```

The logged output should like this:
```
2025-10-20 08:23:33,281 - INFO - Total records: 941634
2025-10-20 08:23:33,281 - INFO - Records with missing coordinates: 212364
2025-10-20 08:23:33,281 - INFO - Failed joins: 0 (0.00%)
2025-10-20 08:23:33,281 - INFO - Join completed in 334.15 seconds
Mismatched records compared to `sr_hex.csv`:
match
True     941631
False         3
Name: count, dtype: int64
```


## Compution vision classification challenge

### Baseline model
The baseline image classifier can be found in [`./notebooks/classification_model_v0.ipynb`](./notebooks/classification_model_v0.ipynb). The model uses a pre-trained `convnext-tiny` backbone from the `timm` library with a custom binary classification head.

### Experiments
Various experiments were performed to test possible improvements to the baseline models. The main experiments are briefly summarised below.

- [`notebooks/classification_model_experiment_01.ipynb`](notebooks/classification_model_experiment_01.ipynb): The aim of this experiment is to measure whether the model accuracy can be improved if we use higher image resolutions. To do this, we will replace the pretrained `convnext_tiny` backbone with a version of `convnext_tiny` that was trained on on 384x384 images instead of the 224x224 images used for our original model. This model (`convnext_tiny.in12k_ft_in1k_384` in `timm`) achieves 1% higher top-1 accuracy on ImageNet than our baseline model.
- [`notebooks/classification_model_experiment_02.ipynb`](notebooks/classification_model_experiment_02.ipynb) The aim of this experiment is to measure whether the model accuracy can be improved by scaling up the model size. To do this, we will replace the pretrained `convnext_tiny.in12k_ft_in1k_384` backbone from experiment 01 with a version of `convnext_small` that was also trained on on 384x384 images. This model (`convnext_small.in12k_ft_in1k_384` in `timm`) achieves 1% higher top-1 accuracy on ImageNet than `convnext_tiny.in12k_ft_in1k_384`. The convnext_small model has 50.2M parameters compared to the 28.6M of convnext_tiny.
- [`notebooks/classification_model_experiment_03.ipynb`](notebooks/classification_model_experiment_01.ipynb) The aim of this experiment is to test a different backbone. To do this, we will replace the pretrained convnext backbone with a version of `CAFormer`. The specific version (`caformer_s36.sail_in22k_ft_in1k_384` in `timm`) achieves 0.7% higher top-1 accuracy on ImageNet than `convnext_small.in12k_ft_in1k_384` while having 10M fewer parameters.

### Evaluation metrics
We will evaluate the following metrics on the validation set:

- [Balanced accuracy score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
- [Average Precision (AP)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- [Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
- [Recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
- [Area Under the Receiver Operating Characteristic Curve (ROC AUC)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)

### Final model
The `CAFormer` model shown significant improvement over the convnext models on most metrics. We will therefore proceed with `caformer_s36.sail_in22k_ft_in1k_384` as the model of choice.

In the final training run, we train the model for more epochs, and add basic data augmentations to prevent overfitting and to expose the model to more diverse training data.

There are two training implementations - one in a notebook, and one in a Python script that generates HTML and Markdown output files.

#### Notebook training
See [`notebooks/classification_model_experiment_04.ipynb`](notebooks/classification_model_experiment_04.ipynb)

#### Python script training
The script can be found in [`scripts/train.py`](scripts/train.py). To run the script, execute the following from the root of the repo:
```bash
python scripts/train.py
```

The script assumes that the image data is available in `./data` in this directory. All outputs are saved in `./output` and model weights are saved in `./models`. If these directories do not exist, they will be created by the script.

The final HTML (and Markdown) output files are available in this repository. Please see [here for the HTML version](./output/final_classifier_summary.html) and [here for the Markdown version](./output/final_classifier_summary.md).


### Hardware requirements
It is recommended to run training on GPUs. All the models can be trained on T4 GPUs like those available in Kaggle and the free tier of Google Colab. Test were specifically run on Kaggle notebooks, and Kaggle-specific instructions are provided in the notebooks.

### Utilities
With the availability of thousands of pre-trained image classification models, it can be challenging to decide which models to use for any given task. It is usually necessary to consider a trade-off between classification accuracy, model size (in terms of number of parameters) and speed. The `timm` library maintains a list of validation and benchmarks results for models in that collection. A notebook is provided that downloads the ImageNet validation results from GitHub, and performs an analysis to determine which models may be strong performers. See [`notebooks/compare_pretrained_models.ipynb`](./notebooks/compare_pretrained_models.ipynb) for details. 
