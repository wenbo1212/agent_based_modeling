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

clear_output()

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
                             
# Set the repetitions, the amount of steps, and the amount of distinct values per variable
replicates = 10
max_steps = 80
distinct_samples = 100

# We get all our samples here
param_values2 = saltelli.sample(problem, distinct_samples,calc_second_order=False)
#print("param_values is:",param_values)

param_values = param_values2

fixed_params={"iim":iim,
                "iem":iem}


# READ NOTE BELOW CODE
batch = BatchRunner(SexDistortion, 
                    max_steps = max_steps,
                    fixed_parameters=fixed_params,
                    variable_parameters = {name:[] for name in problem['names']},
                    model_reporters = model_reporters)

count = 0
data = pd.DataFrame(index=range(replicates*len(param_values)), 
                                columns=[ 'planning','min_distance', 'vision', 'initial_utility','sex_ratio'])
data['Run'], data['gender_ratio'], data['female_count'],data['male_count']  = None, None,None,None

time_start_1 = time.time()
for i in range(replicates):
    for vals in param_values: 
        time_start_2 = time.time()
        # Change parameters that should be integers
        vals = list(vals)
        vals[0] = int(vals[0])
        # Transform to dict with parameter names and their values
        
        variable_parameters = {}
        for name, val in zip(problem['names'], vals):
            variable_parameters[name] = val
        variable_parameters["iim"]=iim
        variable_parameters["iem"]=iem
        batch.run_iteration(variable_parameters, tuple(vals), count)
        iteration_data = batch.get_model_vars_dataframe().iloc[count]
        iteration_data['Run'] = count # Don't know what causes this, but iteration number is not correctly filled
        data.iloc[count, 0:5] = vals
        data.iloc[count, 5:9] = iteration_data
        count += 1

        clear_output()
        progress = count / (len(param_values) * (replicates))
        print(f'{progress * 100:.2f}% done')
        time_end = time.time()
        print ("The program is already running",(time_end -time_start_1),"s" )
        #print('Estimate rest time:',(time_end-time_start_1)/progress - (time_end -time_start_1),'s')



#Sobol analysis result
Si_get_sex_ratio = sobol.analyze(problem, data['gender_ratio'].values, calc_second_order=False, print_to_console=True)

def plot_index(s, params, i, title=''):
    """
    Creates a plot for Sobol sensitivity analysis that shows the contributions
    of each parameter to the global sensitivity.

    Args:
        s (dict): dictionary {'S#': dict, 'S#_conf': dict} of dicts that hold
            the values for a set of parameters
        params (list): the parameters taken from s
        i (str): string that indicates what order the sensitivity is.
        title (str): title for the plot
    """

    if i == '2':
        p = len(params)
        params = list(combinations(params, 2))
        indices = s['S' + i].reshape((p ** 2))
        indices = indices[~np.isnan(indices)]
        errors = s['S' + i + '_conf'].reshape((p ** 2))
        errors = errors[~np.isnan(errors)]
    else:
        indices = s['S' + i]
        errors = s['S' + i + '_conf']
        plt.figure()

    l = len(indices)

    plt.title(title)
    plt.ylim([-0.2, len(indices) - 1 + 0.2])
    plt.yticks(range(l), params)
    plt.errorbar(indices, range(l), xerr=errors, linestyle='None', marker='o')
    plt.axvline(0, c='k')
    


# First order
#print("Si is:",Si)
plot_index(Si_get_sex_ratio, problem['names'], '1', 'First order sensitivity')
plt.show()


# Total order
plot_index(Si_get_sex_ratio, problem['names'], 'T', 'Total order sensitivity')
plt.show()
