# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Programming in Python
# ## Exam: January 20, 2026
#
#
# You can solve the exercises below by using standard Python 3.13 libraries, NumPy, Matplotlib, Pandas, PyMC.
# You can browse the documentation: [Python](https://docs.python.org/3.13/), [NumPy](https://numpy.org/doc/2.3/index.html), [Matplotlib](https://matplotlib.org/3.10.8/users/index.html), [Pandas](https://pandas.pydata.org/pandas-docs/version/2.3/index.html), [PyMC](https://www.pymc.io/projects/docs/en/stable/api.html).
# You can also look at the [slides](https://homes.di.unimi.it/monga/lucidi2425/pyqb00.pdf) or your code on [GitHub](https://github.com).
#
#
# **The exam is "open book", but it is strictly forbidden to communicate with others or "ask questions" online (i.e., stackoverflow is ok if the answer is already there, but you cannot ask a new question or use ChatGPT and similar products). Suspicious canned answers or plagiarism among student solutions will cause the invalidation of the exam for all the people involved.**
#
# To test examples in docstrings use
#
# ```python
# import doctest
# doctest.testmod()
# ```
#
# **SOLVE EACH EXERCISE IN ONE OR MORE NOTEBOOK CELLS AFTER THE QUESTION (delete the `pass` instruction).**
#

# +
import numpy as np
import pandas as pd             # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pymc as pm               # type: ignore
import arviz as az              # type: ignore

# Do not worry about the "FutureWarning"
# -

# ### Exercise 1 (max 3 points)
#
# The file [hibernation.csv](./hibernation.csv) contains data about animal hibernation patterns. The data includes measurements of body mass before and after hibernation, hibernation duration, diet types, and reproductive activity.
#
# Read the dataset into a DataFrame `hib`. Be sure columns hibstart, hibend, hibendyearbefore are correctly interpreted as timestamps (for example, hibstart for row 0 should be interpreted as the 20th of August 2004). The column repro_active should be a boolean. 
#
# Columns:
# - a name: animal identifier
# - year_birth: year the animal was born
# - age: age of the animal during this hibernation period
# - log_age: logarithm (base 10) of age
# - bm_before: body mass before hibernation
# - bm_after: body mass after hibernation
# - hibdur days: hibernation duration in days
# - hibstart: hibernation start date
# - hibend: hibernation end date
# - hibendyearbefore: hibernation end date from previous year
# - bmspring: body mass in spring
# - year: year of observation
# - sex: sex of the animal (m/f)
# - diet: diet type (medium, high fat, protein)
# - age death: age at death
# - repro_active: reproductive activity (yes/no)

hib = pd.read_csv('hibernation.csv', true_values=['yes'], false_values=['no'], 
                  parse_dates=['hibstart', 'hibend', 'hibendyearbefore'], date_format="%d.%m.%y")
hib.head()

assert hib['hibstart'].dtype != str and hib['hibstart'].iloc[0].month == 8
assert hib['repro_active'].dtype == bool

# ### Exercise 2 (max 4 points)
#
# Compute the total number of hibernation observations for males and females separately. Then, for each diet type (medium, high fat, protein), calculate what percentage of observations come from males vs females. To get full marks, do not use explicit loops.

hib.groupby('sex').count()['a name']

100 * hib.groupby(['diet', 'sex']).size() / hib.groupby('diet').size()


# ### Exercise 3 (max 6 points)
#
# Define a function which takes a `pd.Series` (of floats) and computes a new series where each value is the moving average with window size $w$, i.e. the average of the current value and the $w-1$ subsequent values. The resulting series should be $w-1$ elements shorter than the input.
#
# For example, if the series contains values 10., 20., 30., 40., 50., and the window is 3, the result should be a series with 20.0, 30.0, 40.0.
#
# To get full marks, you should declare correctly the type hints (the signature of the function) and add a doctest string with at least to examples.

def moving_average(series: pd.Series, window: int) -> pd.Series:
    """
    Compute the moving (rolling) average, with the given window.  
        
    >>> s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    >>> result = moving_average(s, 3)
    >>> list(result)
    [20.0, 30.0, 40.0]
    >>> s2 = pd.Series([1.0, 2.0, 3.0, 4.0])
    >>> result = moving_average(s2, 2)
    >>> list(result)
    [1.5, 2.5, 3.5]
    """
    r: list[float] = []
    for i in range(len(series)-window+1):
        r.append(series[i:i+window].mean())
    return pd.Series(r)



import doctest
doctest.testmod()

# ### Exercise 4 (max 4 points)
#
# Apply the function defined in Exercise 3 to the data in columns `bm_before` and `bm_after`, both in ascending order, with window size 4. To get full marks, do not use explicit loops.

hib[['bm_before', 'bm_after']].apply(lambda col: moving_average(col.sort_values(), 4), axis=0)

# ### Exercise 5 (max 3 points)
#
# Add a new column called `mass_loss_per_day` that represents the absolute mass lost during hibernation (bm_before - bm_after) per day (hibdur_days). Compute the mean of `mass_loss_per day` for each diet type.

hib['mass_loss_per_day'] = (hib['bm_before'] - hib['bm_after']) / hib['hibdur days']
hib.groupby(['diet'])['mass_loss_per_day'].mean()

# ### Exercise 6 (max 4 points)
#
# Make a scatter plot of days of hibernation vs. age at death. Each point should have a different color for males and females. Put proper labels and a legend.

# +
fig, ax = plt.subplots(figsize=(10, 6))
males = hib[hib['sex'] == 'm']
females = hib[hib['sex'] == 'f']


ax.scatter(males['hibdur days'], males['age death'], alpha=0.6, label='males')
ax.scatter(females['hibdur days'], females['age death'], alpha=0.6, label='females')

ax.set_xlabel('Days of hibernation')
ax.set_ylabel('Age at death')
ax.legend()

fig.tight_layout()
# -

# ### Exercise 7 (max 5 points)
#
# Plot on the same axes the trends of change for each animal ('a name') of the length of hibernation ('hibdur day') along the years in which the data were collected. Use a legend to distinguish the lines. 

# +
fig, ax = plt.subplots(figsize=(10, 10))

for r in hib.groupby('a name')[['hibdur days', 'year']].agg(lambda g: list(g)).itertuples():
    ax.plot(r.year, r._1, marker='o', label=r.Index)

ax.set_xlabel('Year')
ax.set_ylabel('Hibernation days')
ax.legend(title="a name", ncol=5, fontsize='small', bbox_to_anchor=(1, 1))
fig.tight_layout()
# -

# ### Exercise 8 (max 4 points)
#
# Consider only females that are active in reproduction. Sample this statistical model:
#
# - the parameter $\alpha$ is normally distributed with mean 150, and stdev 50
# - the parameter $\beta$ is normally distributed with mean 0, and stdev 10
# - the parameter $\sigma$ is exponentially distributed with $\lambda = 0.1$
# - the mean of the observed value of `bm_after` is given by $\alpha + \beta \cdot D$ where D is the observed value of `hibdur days` (hibernation duration), its std deviation is $\sigma$
#
# Use PyMC to sample the posterior distributions after having seen the actual values for `bm_after`. Plot the posterior with `az.plot_posterior`.

data = hib[(hib['sex'] == 'f') & hib['repro_active']].dropna()
with pm.Model() as model:
    a = pm.Normal('alpha', 150, 50)
    b = pm.Normal('beta', 0, 10)
    s = pm.Exponential('lam', 0.1)

    D = data['hibdur days']

    pm.Normal('bm_after', mu=a + b*D, sigma=s, observed=data['bm_after'])

    idata = pm.sample(seed=6666)


_ = az.plot_posterior(idata)


