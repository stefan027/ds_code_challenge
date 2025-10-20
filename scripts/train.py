"""
Script to train an image classification model on the swimming-pool dataset
and log results. The results are saved as a markdown file and an HTML file for easy viewing.
"""

import sys
from pathlib import Path

# Add the repo root to the Python path
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

import os
from importlib.metadata import version
from typing import Union
import random

from fastai.vision.all import (
    set_seed, DataLoader, DataLoaders, Learner,
    BCEWithLogitsLossFlat, Adam, CSVLogger
)
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms
import albumentations as A
import markdown

from src.data import TiffImageDataset
from src.modeling import (
    create_timm_model, freeze_except_head, unfreeze_all
)
from src.metrics import (
    balanced_accuracy, ap_score, precision, recall, roc_auc
)


IMAGE_DIR = Path("./data/images/swimming-pool")
CLASSES = ["no", "yes"]
POSITIVE_CLASS = "yes"


class TrainingConfig:
    model_name = "final_classifier"

    backbone = "caformer_s36.sail_in22k_ft_in1k_384"
    use_fastai_head = False

    image_size = 384

    num_epoch_finetune_head = 1
    num_epoch_full_finetune = 5
    batch_size = 16
    validation_pct = 0.2
    lr_head = 1e-3
    lr_full = 1e-4

    albumentations_augment = True

cfg = TrainingConfig()


# During training, we will log results to a markdown string `output`
# This markdown string is later saved as a `.html` file for viewing in a browser.
output = """
# Computer Vision classification challenge

## Task description

Use a sample of images from the swimming-pool dataset to develop a model that classifies whether
an image contains a swimming pool or not. Use the provided labels to validate your model.

## Solution overview

This script fine-tunes a 39M parameter [`CAFormer`](https://arxiv.org/abs/2210.13452) image
classification model on the swimming-pool dataset. The dataset consists of aerial images from
Cape Town, and is labelled for the presence of swimming pools or not

A series of experiments were conducted to find the best model architecture, training strategy, and hyperparameters.
Detailed results and analysis of these experiments can be found in the `./notebooks` directory in the repository.

A notebook version of this script is also available as a Jupyter notebook.
Please see `./notebooks/classification_model_experiment_04.ipynb`.

We will evaluate the following metrics on the validation set:\n
- [Balanced accuracy score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
- [Average Precision (AP)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- [Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
- [Recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
- [Area Under the Receiver Operating Characteristic Curve (ROC AUC)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
"""

# ----------------------------------------------
# Package tracking
# ----------------------------------------------
output += """
## Package versions
We keep track of the package versions used for reproducibility. The following key packages were used:\n
"""
pckgs = ["torch", "torchvision", "fastai", "timm"]
for pckg in pckgs:
    output += f"- {pckg}=={version(pckg)}\n"

# ----------------------------------------------
# Reproducibility
# ----------------------------------------------
# Set the random seed for reproducibility. The `set_seed` function sets the seed for `numpy`, `random`, and `torch`.
set_seed(42, reproducible=True)

# ----------------------------------------------
# Load data
# ----------------------------------------------
image_paths = [IMAGE_DIR/f"{category}/{f}" for category in CLASSES for f in os.listdir(IMAGE_DIR/category)]

# Remove non-TIF files
image_paths = [fp for fp in image_paths if fp.suffix.lower() == ".tif"]

output += "\n## Data overview\n"
output += f"Number of image files: {len(image_paths)}\n"

# ----------------------------------------------
# Generate labels and look at class distribution
# ----------------------------------------------
def get_label(fp: Union[str, Path], positive_class: str) -> str:
    """Extracts the label from the file path."""
    label = Path(fp).parts[-2]
    return int(label == positive_class)

# Get the labels for each image file
labels = [get_label(fp, POSITIVE_CLASS) for fp in image_paths]

# Look at class distribution
label_distr = pd.Series(labels).value_counts().to_frame()
label_distr.index.names = ['label']
output += "\n### Class distribution\n"
output += label_distr.to_markdown()

# ----------------------------------------------
# Train/validation split
# ----------------------------------------------
output += """\n
### Training and validation sets
We will use a random 20% of the data as the validation set. That means that 20% of the images will be held out from training and used to evaluate model performance after training.
While there is class imbalance in the dataset, given that we only have two classes and we have a relatively large sample of images, doing a simple random split should result in similarly balanced training and validation splits.
"""

valid_idx = sorted(random.sample(range(len(image_paths)),
                                 k=int(len(image_paths)*cfg.validation_pct)))
train_idx = sorted(list(set(range(len(image_paths))) - set(valid_idx)))
train_fps, train_labels = [image_paths[i] for i in train_idx], [labels[i] for i in train_idx]
valid_fps, valid_labels = [image_paths[i] for i in valid_idx], [labels[i] for i in valid_idx]

output += f"\nTraining/validation split summary:\n\n"
output += f"- Training set size:   {len(train_fps)}\n"
output += f"- Validation set size: {len(valid_fps)}\n"
output += "\nPercentage positive class:\n\n"
output += f"- Training:   {sum(train_labels) / len(train_labels) * 100:.1f}%\n"
output += f"- Validation: {sum(valid_labels) / len(valid_labels) * 100:.1f}%\n"

# ----------------------------------------------
# Look at some sample images
# ----------------------------------------------
sz = (448, 448)
pos = [fp for fp, label in zip(image_paths, labels) if label == 1]
neg = [fp for fp, label in zip(image_paths, labels) if label == 0]
os.makedirs("./output/sample_images", exist_ok=True)
output += "\n### Sample images\n"
for k in range(2):
    i = random.randint(0, len(pos)-1)
    output += f"\n#### Sample image {k+1} - with swimming pool\n"
    output += f"Image path: {pos[i]}\n\n"
    output += f"<img src='sample_images/pos_{i}.png' alt='Swimming Pool - Yes'/>\n"
    Image.open(pos[i]).resize(sz).save(f"./output/sample_images/pos_{i}.png")
for k in range(2):
    i = random.randint(0, len(neg)-1)
    output += f"\n#### Sample image {k+3} - without swimming pool\n"
    output += f"Image path: {neg[i]}\n\n"
    output += f"<img src='sample_images/neg_{i}.png' alt='Swimming Pool - No'/>\n"
    Image.open(neg[i]).resize(sz).save(f"./output/sample_images/neg_{i}.png")

# ----------------------------------------------
# Create `Dataset`s and `DataLoaders`
# ----------------------------------------------
if cfg.albumentations_augment:
    train_transform = A.Compose([
        A.Resize(384, 384),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])
    val_transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])
else:
    train_transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    val_transform = train_transform

output += "\n## Data transformations and augmentations\n"
output += "Image transformations are required to resize the images to the input size expected by the model backbone "
output += "or to reduce the memory footprint during training. Images must also be normalised using the same mean and standard deviation "
output += "as used during pretraining of the model backbone. We will also apply some data augmentations to the training images "
output += "to reduce overfitting and improve model generalisation by exposing the model to more diverse training data.\n\n"
output += "The following transformations and augmentations will be applied to the training images:\n\n"
for tfm in train_transform.transforms:
    output += f"- {tfm.get_class_fullname()}\n"
    for key, value in tfm.get_transform_init_args().items():
        output += f"    - {key}: {value}\n"

# Datasets
ds_trn = TiffImageDataset(paths=train_fps, labels=train_labels, transform=train_transform, use_albumentations=cfg.albumentations_augment)
ds_val = TiffImageDataset(paths=valid_fps, labels=valid_labels, transform=val_transform, use_albumentations=cfg.albumentations_augment)

# Dataloaders
dls_trn = DataLoader(ds_trn, batch_size=cfg.batch_size, shuffle=True)
dls_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)
dls = DataLoaders(dls_trn, dls_val)

# ----------------------------------------------
# Model setup
# ----------------------------------------------
output += "\n## Model setup\n"
output += "We will use a pretrained model from the `timm` library as the backbone of our classifier. "
output += "The final classification head will be randomly initialised. "
output += "For detailed information about the model architecture, see the Appendix.\n\n"
output += f"- Model backbone: `{cfg.backbone}`\n"

model, model_cfg = create_timm_model(cfg.backbone, n_out=1, pretrained=True, use_fastai_head=cfg.use_fastai_head)

# File to which to write the training logs (losses, validation metrics, etc. after each epoch)
log_fp = f'./output/train_log_{cfg.model_name}.csv'
if os.path.exists(log_fp):
    os.remove(log_fp)

# Initialise the `fastai` `Learner`
learn = Learner(
    dls, model, loss_func=BCEWithLogitsLossFlat(), opt_func=Adam,
    metrics=[balanced_accuracy(), ap_score(), precision(), recall(), roc_auc()],
    cbs=[CSVLogger(fname=log_fp, append=True)]
)
learner_summary = learn.summary()
print(learner_summary)

# ----------------------------------------------
# TRAINING
# ----------------------------------------------
output += "\n## Model training\n\n"

# Freeze all model parameters except for the randomly initialised classification head.
# We will first train the classification head, and then fine-tune all model parameters.
if cfg.num_epoch_finetune_head > 0:
    output += "- Freeze all model parameters except for the classification head and "
    output += f"train the head for {cfg.num_epoch_finetune_head} epochs.\n"
    model = freeze_except_head(model)
    # Train the classification head
    learn.fit_one_cycle(cfg.num_epoch_finetune_head, cfg.lr_head)

# Unfreeze all model parameters and fine-tune further
if cfg.num_epoch_full_finetune > 0:
    output += "- Unfreeze all model parameters and "
    output += f"fine-tune the entire model for {cfg.num_epoch_full_finetune} epochs.\n"
    model = unfreeze_all(model)
    learn.fit_one_cycle(cfg.num_epoch_full_finetune, cfg.lr_full)

# Save the model weights for further evaluation and inference:
os.makedirs("./models", exist_ok=True)
torch.save(learn.model.state_dict(), f"./models/model_{cfg.model_name}.pth")

# ----------------------------------------------
# Parse the training log and add to the markdown output
# ----------------------------------------------
with open(log_fp, 'r') as f:
    log_contents = f.readlines()

segments = {1: "log for summary() call"}
if cfg.num_epoch_finetune_head > 0:
    segments[2] = "Training log for fine-tuning the classifier head"
if cfg.num_epoch_full_finetune > 0:
    segments[max(segments)+1] = "Training log for full model fine-tuning"
segment_values = {k: [] for k in segments.keys()}

segment_count = 0
for line in log_contents:
    values = line.strip().split(',')
    if values[0] == 'epoch':
        segment_count += 1
    segment_values[segment_count].append(values)

for k, v in segment_values.items():
    if k == 1:
        continue
    res = pd.DataFrame(segment_values[k][1:], columns=segment_values[k][0])
    res = res.drop(columns=["time"]).astype(float)
    res["epoch"] = res["epoch"].astype(int)
    output += f"\n### {segments[k]}\n\n"
    output += res.to_markdown(index=False) + "\n"

# ----------------------------------------------
# Appendix: detailed model summary
# ----------------------------------------------
output += "\n## Appendix: Detailed model summary\n\n"
output += f"```\n{learner_summary}\n```"

# ----------------------------------------------
# Save the markdown output to a file
# ----------------------------------------------
with open(f"./output/{cfg.model_name}_summary.md", "w") as f:
    f.write(output)

# ----------------------------------------------
# Convert markdown to HTML and save to file
# ----------------------------------------------
html_body = markdown.markdown(output, extensions=['tables', 'fenced_code'])
html_output = f"""
<html>
<head>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/github-markdown-css/github-markdown.min.css">
<style>
body {{
  box-sizing: border-box;
  min-width: 200px;
  max-width: 980px;
  margin: 0 auto;
  padding: 45px;
}}
</style>
</head>
<body class="markdown-body">
{html_body}
</body>
</html>
"""
with open(f"./output/{cfg.model_name}_summary.html", "w") as f:
    f.write(html_output)
