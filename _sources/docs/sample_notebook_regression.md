---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: venv
  language: python
  name: venv
---

# Sample Notebook ( Regression )

This is a sample notebook for a regression type of ML application

```{code-cell} ipython3
from matplotlib import rcParams, cycler
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd
import math
```

### Data Loading

```{code-cell} ipython3
# load and summarize the california housing dataset
from sklearn.datasets import fetch_california_housing
dataset = fetch_california_housing(return_X_y=False, as_frame=True)
```

```{code-cell} ipython3
# checkout the structure and shape of the dataset
dataset.keys()
```

```{code-cell} ipython3
dataset.data.shape
```

### Data Inspection
Look at :
 1. Data distributions 
 2. Basic statistics
 3. Correlations between features and target

```{code-cell} ipython3
# inspect dataset
print('dataset instances : %d' %len(dataset.data))
print('dataset features  : %s' %len(dataset.feature_names))
print('dataset atributes : %s' %dataset.feature_names)
print('dataset feature   : %s' %dataset.target_names)
```

```{code-cell} ipython3
X = dataset.data
y = dataset.target
```

```{code-cell} ipython3
# Data Distributions
X.hist(bins=80, figsize=(15, 15), grid=False);
```

```{code-cell} ipython3
# Basic statistics
X.describe()
```

From the histograms and basic statistics we can already see here 2 important points regarding the dataset:

 1. Basic Statistics : some of the attributes contain outliers, like AveRooms and AveBedrms  
 2. Distributions : the scales of the attributes are quite different


 Conclusions:

 1. Some outliers treatment is necessary (removal of outliers for example)
 2. Some data standarization is also necessary to bring all features into an equivalent scale . This is done to avoid the variance scale from a large feature to dominate and bias the model.

```{code-cell} ipython3
# Correlations between feature and target
import seaborn as sns
corr = dataset.data.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5);
```

```{code-cell} ipython3
# outlier detection using covariance
from sklearn.covariance import EllipticEnvelope
cov = EllipticEnvelope(random_state=0).fit(X)
# check outliers on each attribute : predict returns 1 for an inlier and -1 for an outlier
covariances=cov.predict(X)
outliers=[i for i in range(len(covariances)) if covariances[i] == -1]
print('found : %2d outliers in data' %len(outliers))
```

We can already see there is a stronger correlation between Average Rooms for example and Mean House Value.
We could use this in order to select stronger features for training.

+++

### Model Training

```{code-cell} ipython3
# apply transformations to the attrributes and target
from sklearn.preprocessing import *
X_scaled = MinMaxScaler().fit_transform(X) 
y_scaled = MinMaxScaler().fit_transform(y.values.reshape(-1, 1))
```

```{code-cell} ipython3
# evaluate several regression algorithms on the dataset and create several models
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

# define a dictionary with the algorithms we would like to test out on the data
# Note : model hyperparameters here are not tuned!
models = {
    'LinearRegression' : LinearRegression(),
    'ElasticNet' : ElasticNet(alpha=1.0, l1_ratio=0.5),
    'RandomForestRegressor' : RandomForestRegressor(n_estimators=10),
}

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
for name,model in models.items():
    # evaluate model
    scores = cross_val_score(model, X_scaled, y_scaled, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # force scores to be positive
    scores = abs(scores)
    print('Model : %-24s Mean MAE: %.3f (%.3f)' % (name , scores.mean(), scores.std()))
```

### Model Predict

```{code-cell} ipython3
# Select the best model and make a prediction with it
selected_model='RandomForestRegressor'
model = models[selected_model]
# fit model
model.fit(X_scaled, y_scaled)
# define new data to predict the value of the house
new_housing_data = [8.32,41,6.98,1,322,2.55,37.88,-122]
scaler=MinMaxScaler().fit(np.array(new_housing_data).reshape(-1,1))
new_housing_data_scaled = scaler.transform(np.array(new_housing_data).reshape(-1,1))
# make a prediction
y_predicted = model.predict(new_housing_data_scaled.T)
y_scaler = MinMaxScaler().fit(y.values.reshape(-1,1))
# summarize prediction , inverse back the scaled prediction
print('House Predicted Value (KDollar): %.3f' %y_scaler.inverse_transform(y_predicted.reshape(-1,1)))
```
