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
# ## Exam: January 21, 2025
#
# You can solve the exercises below by using standard Python 3.12 libraries, NumPy, Matplotlib, Pandas, PyMC.
# You can browse the documentation: [Python](https://docs.python.org/3.12/), [NumPy](https://numpy.org/doc/1.26/index.html), [Matplotlib](https://matplotlib.org/3.10.0/users/index.html), [Pandas](https://pandas.pydata.org/pandas-docs/version/2.2/index.html), [PyMC](https://www.pymc.io/projects/docs/en/stable/api.html).
# You can also look at the [slides](https://homes.di.unimi.it/monga/lucidi2425/pyqb00.pdf) or your code on [GitHub](https://github.com).
#
# **It is forbidden to communicate with others or "ask questions" online (i.e., stackoverflow is ok if the answer is already there, but you cannot ask a new question or use ChatGPT and similar products)**
#
# To test examples in docstrings use
#
# ```python
# import doctest
# doctest.testmod()
# ```
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

hib = pd.read_csv('hibernation.csv', 
                  parse_dates=['hibstart', 'hibend', 'hibendyearbefore'],
                  dayfirst=True)
hib['repro_active'] = hib['repro_active'] == 'yes'


# ### Exercise 2 (max 4 points)
#
# Compute the total number of hibernation observations for males and females separately. Then, for each diet type (medium, high fat, protein), calculate what percentage of observations come from males vs females. To get full marks, do not use explicit loops.

# Numero totale di osservazioni per maschi e femmine
total_by_sex = hib['sex'].value_counts()
print("Numero totale di osservazioni per sesso:")
print(total_by_sex)

# Percentuale di maschi vs femmine per ogni tipo di dieta
diet_sex_counts = hib.groupby(['diet', 'sex']).size()
diet_totals = hib.groupby('diet').size()
diet_sex_percentages = (diet_sex_counts / diet_totals.loc[diet_sex_counts.index.get_level_values('diet')].values) * 100
print("\nPercentuale di maschi vs femmine per tipo di dieta:")
print(diet_sex_percentages)


# ### Exercise 3 (max 7 points)
#
# Define a function which takes a `pd.Series` (of floats) and computes a new series where each value is the average of the current value and the two subsequent values (moving average of window size 3). The resulting series should be 2 elements shorter than the input.
#
# For example, if the series contains values 10, 20, 30, 40, 50, the result should be a series with 20.0, 30.0, 40.0.
#
# To get full marks, you should declare correctly the type hints (the signature of the function) and add a doctest string.

def moving_average_3(series: pd.Series) -> pd.Series:
    """
    Calcola la media mobile di finestra 3 su una Series di float.
    
    Ogni valore del risultato è la media del valore corrente e dei due successivi.
    La serie risultante è 2 elementi più corta dell'input.
    
    Args:
        series: Una pd.Series di valori float
        
    Returns:
        Una pd.Series con le medie mobili, 2 elementi più corta dell'input
        
    >>> s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    >>> result = moving_average_3(s)
    >>> list(result)
    [20.0, 30.0, 40.0]
    >>> s2 = pd.Series([1.0, 2.0, 3.0, 4.0])
    >>> result2 = moving_average_3(s2)
    >>> list(result2)
    [2.0, 3.0]
    """
    return series.rolling(window=3).mean()[2:]


import doctest
doctest.testmod()

# ### Exercise 4 (max 2 points)
#
# Apply the function defined in Exercise 3 to the column `bm_before`. To get full marks, do not use explicit loops.
hib['bm_before_ma3'] = moving_average_3(hib['bm_before'])



# ### Exercise 5 (max 5 points)
#
# Make a copy of the DataFrame `hib` in which you add a new column called `mass_loss` that represents the absolute mass lost during hibernation (bm_before - bm_after). Then, add another column called `mass_loss_percent` that represents the percentage of body mass lost relative to the pre-hibernation mass. Finally, filter the copy to keep only rows where the animal was reproductively active (repro_active == 'yes').
hib2 = hib.copy()

hib2['mass_loss'] = hib2['bm_before'] - hib2['bm_after']
hib2['mass_loss_percent'] = hib2['mass_loss'] / hib2['bm_before'] * 100

hib2 = hib2[hib2['repro_active']]



# ### Exercise 6 (max 4 points)
#
# Add to the DataFrame `hib` a column with the standardized value of `hibdur days` (hibernation duration). Remember that the standardized value measures how many standard deviations a specific value is far from the mean. If you have an ndarray of values `xx`: `(xx - xx.mean())/xx.std()`. Then plot a density histogram of this new column.

# Calcolo dei valori standardizzati della durata dell'ibernazione
hibdur_values = hib['hibdur days'].values
hib['hibdur_standardized'] = (hibdur_values - hibdur_values.mean()) / hibdur_values.std()

# Plot dell'istogramma di densità
plt.figure(figsize=(10, 6))
plt.hist(hib['hibdur_standardized'], bins=30, density=True, alpha=0.7, edgecolor='black')
plt.xlabel('Durata ibernazione standardizzata')
plt.ylabel('Densità')
plt.title('Istogramma di densità della durata dell\'ibernazione standardizzata')
plt.grid(True, alpha=0.3)
plt.show()


# ### Exercise 7 (max 4 points)
#
# Plot a matrix of scatter plots (for each pair a,b you can plot just a,b and leave b,a empty) of all the combinations of `bm_before`, `bm_after`, `hibdur days`, and `age`. They should appear all in the same figure. Put also a proper title to each plot.
vars_to_plot = ['bm_before', 'bm_after', 'hibdur days', 'age']
n = len(vars_to_plot)

fig, axes = plt.subplots(n, n, figsize=(10, 10))

for i in range(n):
    for j in range(n):
        if i < j:
            axes[i, j].scatter(hib[vars_to_plot[j]], hib[vars_to_plot[i]], alpha=0.5)
            axes[i, j].set_xlabel(vars_to_plot[j])
            axes[i, j].set_ylabel(vars_to_plot[i])
            axes[i, j].set_title(f'{vars_to_plot[i]} vs {vars_to_plot[j]}')
        else:
            axes[i, j].axis('off')

plt.tight_layout()
plt.show()



# ### Exercise 8 (max 5 points)
#
# Consider this statistical model:
#
# - the parameter $\alpha$ is normally distributed with mean 150, and stdev 50
# - the parameter $\beta$ is normally distributed with mean 0, and stdev 10
# - the parameter $\sigma$ is exponentially distributed with $\lambda = 0.1$
# - the mean of the observed value of `bm_after` is given by $\alpha + \beta \cdot D$ where D is the observed value of `hibdur days` (hibernation duration), its std deviation is $\sigma$
#
# Use PyMC to sample the posterior distributions after having seen the actual values for `bm_after`. Plot the posterior.
D = hib['hibdur days'].values
Y = hib['bm_after'].values

with pm.Model() as model:
    
    alpha = pm.Normal('alpha', mu=150, sigma=50)
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.Exponential('sigma', lam=0.1)
    
    mu = alpha + beta * D
    
    bm_obs = pm.Normal('bm_obs', mu=mu, sigma=sigma, observed=Y)
    
    trace = pm.sample(2000, tune=1000, chains=2, target_accept=0.9)

# Scatter plot: giorni di ibernazione vs età alla morte, colorato per sesso
fig, ax = plt.subplots(figsize=(10, 6))

# Filtra i dati per rimuovere valori NaN in age_death
hib_valid = hib.dropna(subset=['age death'])

# Mappa sesso a colori: m -> blue, f -> red
colors = hib_valid['sex'].map({'m': 'blue', 'f': 'red'})

# Un'unica chiamata a scatter
scatter = ax.scatter(hib_valid['hibdur days'], hib_valid['age death'], 
                     c=colors, alpha=0.6, s=50)

# Crea manualmente la legenda
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', alpha=0.6, label='Maschi'),
                   Patch(facecolor='red', alpha=0.6, label='Femmine')]
ax.legend(handles=legend_elements)

# Etichette e titolo
ax.set_xlabel('Durata ibernazione (giorni)')
ax.set_ylabel('Età alla morte (anni)')
ax.set_title('Durata ibernazione vs Età alla morte per sesso')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
