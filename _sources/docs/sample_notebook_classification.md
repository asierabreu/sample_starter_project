---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Overview

The task at hand is classification of terrain types from satellite images.
We will use two different approaches:
- A "standard" transfer learning approach where we will build a CNN using a base model (from Imagenet)
- A "bayesian" approach where we will take into account uncertainty on the provided labels

## Workflow

1. [Data Inspection](#inspection) 
    - Loading
    - Inspection
    - Preprocessing
2. [Modeling](#model-definition)
    - Convolutional Neural Network
    - Bayesian Neural Network
3. [Prediction](#prediction)

```{code-cell} ipython3
# Software install (as required)
#!pip install -r ../requirements.txt
```

```{code-cell} ipython3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tensorflow_addons as tfa
```

## Data Inspection <a name="inspection"></a>

```{code-cell} ipython3
# Data Loading : load the EuroSat dataset
# 27000 Sentinel-2 satellite images covering 13 spectral bands. 
# Reference : https://github.com/phelber/eurosat

# load train, test & validation splits into 60%, 20%,20% respectively
(ds_train, ds_test, ds_valid), ds_info   = tfds.load(
    "eurosat", 
    split=["train[:60%]","train[60%:80%]","train[80%:]"],
    as_supervised=True,
    shuffle_files=True,
    with_info=True)
```

```{code-cell} ipython3
ds_info.features
```

```{code-cell} ipython3
# Basic Info
class_names = ds_info.features["label"].names
num_classes = ds_info.features["label"].num_classes
image_size  = ds_info.features["image"]
print('Total no. of classes : %d' %num_classes)
print('Class labels  : %s' %class_names)
print("Total examples: %d" %(len(ds_valid)+len(ds_train)+len(ds_test)))
print("Train set size: %d" %len(ds_train)) 
print("Test set size : %d" %len(ds_test))   
print("Valid set size: %d" %len(ds_valid))
print("")
ds = ds_train.take(1)  # Only take a single example
for image, label in ds: 
  print('image tensor shape: %s' %image.shape)
  print('label tensor type: %s' %label)
```

```{code-cell} ipython3
# show a few examples from the train dataset
tfds.as_dataframe(ds_train.take(10), ds_info)
```

```{code-cell} ipython3
# Class balance check : is the dataset imbalanced?
fig, ax = plt.subplots(1, 1, figsize=(10,6))

labels, counts = np.unique(np.fromiter(ds_train.map(lambda x, y: y), np.int32), 
                       return_counts=True)
ax.set_xlabel('Counts')
ax.grid(True,ls='--')
ax.set_title("Counts by type of terrain");
sns.barplot(x=counts, y=[class_names[l] for l in labels], label="Total")
sns.despine(left=True, bottom=True)
```

```{code-cell} ipython3
def prepare_for_training(ds, cache=True, batch_size=64, shuffle_buffer_size=1000):
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()
  # one-hot encode labels
  ds = ds.map(lambda d: (d["image"], tf.one_hot(d["label"], num_classes)))
  # shuffle the dataset
  ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  # Repeat forever
  ds = ds.repeat()
  # split to batches
  ds = ds.batch(batch_size)
  # `prefetch` lets the dataset fetch batches in the background while the model is training.
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return ds
```

```{code-cell} ipython3
batch_size = 64
# preprocess training & validation sets
ds_train = prepare_for_training(ds_train, batch_size=batch_size)
ds_valid = prepare_for_training(ds_valid, batch_size=batch_size)
```

### Model Definition <a name="model definition"></a>

  1. We use a base model (pretrained neural net for the imagenet challenge and specify that is trainable
  2. We add a top model with a softmax classification layer 

```{code-cell} ipython3
# BASE MODEL
model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/feature_vector/2"
# download & load the layer as a feature vector
base_model = hub.KerasLayer(model_url, output_shape=[1280], trainable=True, name='base_layer')
```

```{code-cell} ipython3
model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.Dense(num_classes, activation="softmax", name="classification_layer")
])
# build the model with input image shape as (64, 64, 3)
model.build([None, 64, 64, 3])
model.compile(
    loss="categorical_crossentropy", 
    optimizer="adam", 
    metrics=["accuracy", tfa.metrics.F1Score(num_classes)]
)
```

```{code-cell} ipython3
model.summary()
```

```{code-cell} ipython3
model_name = "satellite-classification"
model_path = os.path.join("../models", model_name + ".h5")
if not os.path.exists("../models"):
    os.makedirs(model_path)
```

```{code-cell} ipython3
# set the training & validation steps since we're using .repeat() on our dataset
# number of training steps
n_training_steps   = int(num_examples * 0.6) // batch_size
# number of validation steps
n_validation_steps = int(num_examples * 0.2) // batch_size
```

### Model Training <a name="model training"></a>

```{code-cell} ipython3
verbose=1
epochs=5
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, verbose=1)
# train the model
history = model.fit(
    train_ds, validation_data=valid_ds,
    steps_per_epoch=n_training_steps,
    validation_steps=n_validation_steps,
    verbose=verbose, epochs=epochs, 
    callbacks=[model_checkpoint]
)
```

### Model Evaluation <a name="model evaluation"></a>

+++
