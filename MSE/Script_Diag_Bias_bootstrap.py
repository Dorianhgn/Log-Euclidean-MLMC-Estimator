"""
Author: Dorian Hugonnet & Lise Pihan
File: Script_Diag_Bias
Objective: Plot the bias of the variance Multi-Level (ML) & Variance Log Multi Level (LML)
"""

# Lib imporation :
#import numpy as np
import cupy as np # Oui je sais c'est une abomination mais c'est juste pour pas à tout rechanger si on change de PC
import numpy as npp
import matplotlib.pyplot as plt
from tqdm import tqdm
np.fuse(npp.float32)

from datetime import datetime

import json
import os 

# Obtenir la date et l'heure actuelles
now = datetime.now()
date_time = now.strftime("%Y-%m-%d_%H-%M-%S")


# Variable & functions def: 
def f_vec(X,v):
    return v.T@X

def f_vec_squared(X,v):
    return (v.T@X)**2

d = 10 # lenght of random vectors

v_1= np.array([-0.00606533, -0.02837768, -0.20481078, -0.05524456,  0.00408442, -0.02378791, -0.11289296, -0.09047946, -0.0828985,   0.01015773])
v_0 = np.array([-0.04775578, -0.02158142, -0.22861181,  0.00999135 , 0.05261623, -0.03810132, -0.08892537, -0.09858141, -0.06851002,  0.09957945])
exp_var = np.sum(np.square(v_1))


# Function that computes the estimator of the variance in ML & LML :
def estim_var_MLMC(N, d, M_0, M_1, batch_size):
    var_results = []
    for i in range(0, N, batch_size):
        actual_batch_size = min(batch_size, N - i)
        X1 = np.random.standard_normal((actual_batch_size, d, M_0))
        X0 = np.random.standard_normal((actual_batch_size, d, M_1))

        Y0_0_squared = f_vec_squared(X0, v_0)
        Y1_0_squared = f_vec_squared(X1, v_0)
        Y1_1_squared = f_vec_squared(X1, v_1)
        Var_MLMC_batch = np.mean(Y0_0_squared, axis=1) + np.mean(Y1_1_squared - Y1_0_squared, axis=1)
        
        var_results.extend(Var_MLMC_batch)

        # Free memory
        X1, X0 = None, None
        np.get_default_memory_pool().free_all_blocks()

    return np.array(var_results)

def estim_var_MLMC_log(N, d, M_0, M_1, batch_size):
    var_results = []
    for i in range(0, N, batch_size):
        actual_batch_size = min(batch_size, N - i)
        X1 = np.random.standard_normal((actual_batch_size, d, M_0))
        X0 = np.random.standard_normal((actual_batch_size, d, M_1))

        Y0_0_squared = f_vec_squared(X0, v_0)
        Y1_0_squared = f_vec_squared(X1, v_0)
        Y1_1_squared = f_vec_squared(X1, v_1)
        Var_MLMC_log_batch = np.exp(np.log(np.mean(Y0_0_squared, axis=1)) + np.log(np.mean(Y1_1_squared, axis=1))-np.log(np.mean(Y1_0_squared, axis=1)))
    
        var_results.extend(Var_MLMC_log_batch)

        # Free memory if necessary
        X1, X0 = None, None
        np.get_default_memory_pool().free_all_blocks()

    return np.array(var_results)


# Function that computes the sample size of each level for a given budget eta
def M_ell_tab(eta, V_ell, C_ell,m=0):
    """
    eta : budget
    V_ell : tableau contenant [V_0,V_1,...,V_L]
    C_ell : tableau contenant [C_0,C_1,...,C_L]

    return : la formule M_ell au dessus qui nous donne le nombre d'élement optimal à simuler
    """
    eta_tilde = eta # - m*np.sum(...)
    S_L = np.sum(np.sqrt(np.multiply(V_ell,C_ell)))

    return m + 1 + np.floor(
        (eta_tilde/S_L)*np.sqrt(V_ell*1/C_ell)
    )


# Since we can't compute all at once due to M_0 and M_1 increasing, we need to separate the caluclations by computing batch_size first
def get_max_batch_size(d, M_0, M_1, free_memory, memory_utilization=0.8):
    """
    Calculate the maximum batch size directly based on the memory usage per data sample and the available memory.
    """
    # Calculate the memory usage per sample
    bytes_per_number = np.dtype('float32').itemsize
    omega = (d+1) * bytes_per_number * (M_0 + M_1)  # Total memory used per sample

    # Calculate the maximum amount of memory available for batches
    max_memory_per_batch = free_memory * memory_utilization

    # Calculate the maximum batch size that can fit within the memory limit
    batch_size = int(max_memory_per_batch / omega)  # Use int to ensure we get a whole number

    return batch_size


# Define a function to perform the bootstrap and calculate the bias.
def bootstrap_bias(var_samples, num_samples=10000):
    """
    This function computes the bias using the bootstrap method.
    
    Parameters:
    - var_samples: numpy array, the original sample of variance estimates
    - exp_var: float, the expected variance to compare against
    - num_samples: int, the number of bootstrap samples to generate
    
    Returns:
    - numpy array, contening the bootstrap means
    """
    # Resampling with replacement to create bootstrap samples
    bootstrap_means = np.empty(num_samples)
    n = len(var_samples)
    for i in range(num_samples): # Loop to get num_samples of variances from var_samples taking randomly
        sample_indices = np.random.randint(0, n, n)  # Sample randomly the whole sample (taking n values from var_samples randomly so some indices could repeat)
        bootstrap_sample = var_samples[sample_indices] # For example, taking bootstrap_sample = var_samples[10], var_samples[1], var_samples[10], ..., var_samples[153] with lengh n
        bootstrap_means[i] = np.mean(bootstrap_sample) 

    return np.mean(bootstrap_means), np.var(bootstrap_means)


# Computation of our V_ell with N_pilote samples
N_pilote = 10000
X = np.random.standard_normal((d,N_pilote))
V_0 = np.var(f_vec_squared(X,v_0))
V_1 = np.var(f_vec_squared(X,v_1) - f_vec_squared(X,v_0))

V_ell = np.array([V_0,V_1])
C_ell = np.array([0.75,1.])  # Couts de nos estimateurs

N=10000  # On génère à chaque fois N estimations de nos variances V_ML et V_LML pour calculer leur variance et leur biais
budget_eta = npp.logspace(1,4,10)  # Budget qui varie entre 10^1 et 10^4 

mean_tab_final_val = []
var_tab_final_val = []
var_MLMC_tab = []
mean_log_tab_final_val = []
var_log_tab_final_val = []
var_MLMC_log_tab = []

# Get the current free memory on CUDA device
free_memory = np.cuda.Device().mem_info[0]

# Main loop
# Directory where the file will be saved
directory = 'bias_figsave'

# Ensure the directory exists
if not os.path.exists(directory):
    os.makedirs(directory)

# Set the temporary filename within the specified directory
temp_filename = os.path.join(directory, 'computation_progress_temp.json')

# Open a file to save the progress
with open(temp_filename, 'w') as file:
    # Write headers or initial data structure setup if necessary
    results = []

    for eta in tqdm(budget_eta):
        # Calculate M_0 and M_1 for the current budget
        M_0, M_1 = map(int, M_ell_tab(eta, V_ell, C_ell))

        # Compute the batch size
        batch_size = get_max_batch_size(d, M_0, M_1, free_memory)
        if batch_size <= 10:
            break  # Exit if batch size is too small

        # Calculate estimators for MLMC
        Tab_var_MLMC = estim_var_MLMC(N, d, M_0, M_1, batch_size)
        # Calculate var and append it in the tab
        var_MLMC = np.var(Tab_var_MLMC)
        var_MLMC_tab.append(var_MLMC)
        # Calculate the bootstrap bias and append it in the tab
        Mean_MLMC_bootstrap, Var_MLMC_bootstrap = bootstrap_bias(Tab_var_MLMC)
        mean_tab_final_val.append(Mean_MLMC_bootstrap)
        var_tab_final_val.append(Var_MLMC_bootstrap)

        Tab_var_MLMC = None
        np.get_default_memory_pool().free_all_blocks()

        # Calculate estimators for MLMC log
        Tab_var_MLMC_log = estim_var_MLMC_log(N, d, M_0, M_1, batch_size)
        # Calculate var and append it in the tab
        var_MLMC_log = np.var(Tab_var_MLMC_log)
        var_MLMC_log_tab.append(var_MLMC_log)
        # Calculate the bootstrap bias and append it in the tab
        Mean_MLMC_log_bootstrap, Var_MLMC_log_bootstrap = bootstrap_bias(Tab_var_MLMC_log)
        mean_log_tab_final_val.append(Mean_MLMC_log_bootstrap)
        var_log_tab_final_val.append(Var_MLMC_log_bootstrap)


        Tab_var_MLMC_log = None
        np.get_default_memory_pool().free_all_blocks()

        # Append results to list and write to file
        results.append({
            "eta": eta,
            "M_0": M_0,
            "M_1": M_1,
            "batch_size": batch_size,
            "var_MLMC": var_MLMC.item(),
            "var_MLMC_log": var_MLMC_log.item(),
        })
        file.write(json.dumps(results[-1]) + "\n")  # Write the latest result as a new line in JSON format
        file.flush()

# After the loop, generate the new filename with datetime
new_filename = f'computation_progress_{date_time}.json'

# Rename the file
os.rename(temp_filename, new_filename)

# 

mean_tab_final_val= [x.item() for x in mean_tab_final_val]
var_tab_final_val= [x.item() for x in var_tab_final_val]

mean_log_tab_final_val= [x.item() for x in mean_log_tab_final_val]
var_log_tab_final_val= [x.item() for x in var_log_tab_final_val]


# Plot log-log
plt.figure(figsize=(10, 6))

# Plot des données
#plt.loglog(budget_eta, mean_tab_final_val, marker='o', label='Biais carré Tab')
plt.loglog(budget_eta, var_tab_final_val, marker='s', label='Var bootstrap samples')
#plt.loglog(budget_eta, mean_log_tab_final_val, marker='^', label='Biais carré Log Tab', c='g')
plt.loglog(budget_eta, var_log_tab_final_val, marker='x', label='Var bootstrap samples')

# Ajouter titre et légendes
plt.title(f'Graphique Log-Log pour n={N} using Bootstrap')
plt.xlabel('Budget eta')
plt.ylabel('Valeurs')
plt.legend()


# Configurer la grille
plt.grid(True)

# Afficher le plot
#plt.show()

# Sauvegarder le plot avec la date et l'heure dans le nom du fichier
plt.savefig(f'bias_figsave/graph_boostrap_n={N}_{date_time}.png')


# Fermer la figure pour libérer la mémoire
plt.close()