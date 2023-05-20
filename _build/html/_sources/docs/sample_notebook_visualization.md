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
  name: python3
---

# Sample Notebook ( Visualization )

This is a sample notebook for visualization of complex data

```{code-cell} ipython3

import matplotlib.pyplot as plt
import seaborn
import matplotlib
import numpy as np
import pandas as pd
import random
```

```{code-cell} ipython3
# Client Request Description : 
# I have data for the accuracy scores of 5 algorithms.
# However, these scores have been measured in regard to different groups for each algorithm. 
# For example, Algorithm 1 has 10 different scores one for each group from 1 to 10, while Algorithm 2 has just 1 score measured with regard to one group.
```

```{code-cell} ipython3
# create a Python dictionary with different array lengths
accuracies_dict = dict(algo1=np.array([random.random() for i in range(10)]),
                       algo2=random.random(),
                       algo3=np.array([random.random() for i in range(6)]),
                       algo4=np.array([random.random() for i in range(2)]),
                       algo5=np.array([random.random() for i in range(8)]),
                    )
```

```{code-cell} ipython3
# create a Python dictionary to help up structure the data and present it
df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in accuracies_dict.items()]))
```

```{code-cell} ipython3
df
```

```{code-cell} ipython3
# font asthetics
def set_font():
    font = {'family' : 'Palatino',
            'weight' : 'bold',
            'size'   : 16}
    axes = {'linewidth':1.5}

    matplotlib.rc('font', **font)
    matplotlib.rc('axes', **axes)
    #matplotlib.rc('text', usetex=True
```

```{code-cell} ipython3
set_font()
```

```{code-cell} ipython3
f,axs = plt.subplots(2,1,figsize=(15,15))

# matplotlib default colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# Plot the accuracy of each algorithm over the different groups
df.plot(kind='bar',ax=axs[0])
axs[0].set_xlabel('group id')
axs[0].set_ylabel('accuracy')
axs[0].set_title('Accuracy of each algorithm over the different groups');
axs[0].set_ylim([0,1])

# Plot the AVERAGE algorithm accuracy over the different groups
df.apply(lambda row : row.mean()).plot(kind='bar',ax=axs[1],color=colors)
axs[1].set_ylabel('accuracy')
axs[1].set_title('AVERAGE algorithm accuracy over the different groups');
axs[1].set_ylim([0,1])

plt.xticks(rotation=0);
```

```{code-cell} ipython3

```
