from IPython.display import clear_output
import mesa
import scipy.optimize as opt
import math
import numpy as np
from scipy.optimize import Bounds
from functools import reduce
from mesa.batchrunner import BatchRunner
import os
import time
from sex_distortion import SexDistortion
import sex_distortion as sd
import SALib
from SALib.sample import saltelli
from SALib.analyze import sobol
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
clear_output(wait=True)
print("Everything imported!")

iim_param=[0.01394223,1.07057479]
iem_param=[ 0.28925115,1.43600609,-0.19679485]

#income_invest_model
iim=lambda income: 1/(iim_param[0]*income/100+iim_param[1])
#invest_edu_model
def iem(invest,gender):
    if(invest/100+1<0):
        print(invest)
    x=np.log(invest/100+1)
    return 1/(np.exp(-iem_param[0]*(x+iem_param[1]))+1)+iem_param[2]
    
   
    
problem = {
    'num_vars': 5,
    'names': [ 'planning','min_distance', 'vision', 'initial_utility','sex_ratio'],
    'bounds': [ [2,4],      [0.2,0.8],     [1,1.7] ,   [0.5,1.5],     [0.8,1.2]]}

# Set the outputs
model_reporters={"male_income": sd.compute_male_income,
                              "female_income":sd.compute_female_income,
                              "male_count":sd.male_at_birth,
                              "female_count":sd.female_at_birth,
                              "avg_children":sd.avg_children,
                              "pressure":sd.p_pressure,
                              "gender_ratio": sd.get_sex_ratio
                             }
                             
data = {}
replicates =10
max_steps = 80
distinct_samples = 10

fixed_params={"iim":iim,
                "iem":iem}

for i, var in enumerate(problem['names']):
    # Get the bounds for this variable and get <distinct_samples> samples within this space (uniform)
    samples = np.linspace(*problem['bounds'][i], num=distinct_samples)
    
    if var == 'planning':
        samples = np.linspace(*problem['bounds'][i], num=distinct_samples, dtype=int)
        print("samples :", samples)
    
    batch = BatchRunner(SexDistortion, 
                        max_steps=max_steps,
                        iterations=replicates,
                        fixed_parameters=fixed_params,
                        variable_parameters={var: samples},
                        model_reporters=model_reporters,
                        display_progress=True)
    
    batch.run_all()
    
    data[var] = batch.get_model_vars_dataframe()
    pd.DataFrame(data[var]).to_csv('sa/OFAT.csv')

    
    
def plot_param_var_conf(ax, df, var, param, i):
    """
    Helper function for plot_all_vars. Plots the individual parameter vs
    variables passed.

    Args:
        ax: the axis to plot to
        df: dataframe that holds the data to be plotted
        var: variables to be taken from the dataframe
        param: which output variable to plot
    """
    x = df.groupby(var).mean().reset_index()[var]
    y = df.groupby(var).mean()[param]

    replicates = df.groupby(var)[param].count()
    err = (1.96 * df.groupby(var)[param].std()) / np.sqrt(replicates)

    
    ax.plot(x, y, c='k')
    ax.fill_between(x, y - err, y + err)

    ax.set_xlabel(var)
    ax.set_ylabel(param)
        
        
def plot_all_vars(df, param):
    """
    Plots the parameters passed vs each of the output variables.

    Args:
        df: dataframe that holds all data
        param: the parameter to be plotted
    """

    f, axs = plt.subplots(6, figsize=(7, 10))
    
    for i, var in enumerate(problem['names']):
        plot_param_var_conf(axs[i], data[var], var, param, i)


for param in {'gender_ratio'}:
    plot_all_vars(data, param)
    plt.show()
