"""
Author: Dorian Hugonnet & Lise Pihan
File: Script_Diag_Bias
Objective: Plot the bias of the variance Multi-Level (ML) & Variance Log Multi Level (LML)
"""

# Lib imporation :
#import numpy as cp
import cupy as cp # Oui je sais c'est une abomination mais c'est juste pour pas à tout rechanger si on change de PC
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
cp.fuse(np.float32)

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

# TRUCS A CHANGER
# v_1= cp.array([-0.00606533, -0.02837768, -0.20481078, -0.05524456,  0.00408442, -0.02378791, -0.11289296, -0.09047946, -0.0828985,   0.01015773])
# v_0 = cp.array([-0.04775578, -0.02158142, -0.22861181,  0.00999135 , 0.05261623, -0.03810132, -0.08892537, -0.09858141, -0.06851002,  0.09957945])
# v_array = [v_0,v_1]
# exp_var = cp.sum(cp.square(v_1))
# C_ell = cp.array([0.75,1.])  # Couts de nos estimateurs

N=100000  # On génère à chaque fois N estimations de nos variances V_ML et V_LML pour calculer leur variance et leur biais
budget_eta = np.logspace(1,4,10)  # Budget qui varie entre 10^1 et 10^4 

v= cp.array([-0.00606533, -0.02837768, -0.20481078, -0.05524456,  0.00408442, -0.02378791, -0.11289296, -0.09047946, -0.0828985,   0.01015773]) # Real life
v_0 = cp.array([-0.26251362, -0.22397083, -0.28459696, -0.14160629,  0.11507459,
       -0.01314795,  0.00368215, -0.2233519 , -0.0494188 , -0.09833207])
v_1 = cp.array([-0.09152057,  0.26501426, -0.26361748, -0.09584528,  0.07564258,
       -0.28932995, -0.15172387, -0.17311338, -0.02250007, -0.02662765])
v_2 = cp.array([ 0.02806491,  0.18164134, -0.02569311,  0.08406244,  0.09760685,
       -0.2754576 , -0.18692242,  0.05429335, -0.05959692, -0.16104073])
v_3 = cp.array([ 0.00719622, -0.01367537, -0.04378771,  0.15642576,  0.03295938,
       -0.1364489 , -0.02709714, -0.16822205, -0.15617831, -0.05832736])

v_array = [v_0, v_1, v_2, v_3]
exp_var = cp.sum(cp.square(v))
def expected_var(v):
    return cp.sum(cp.square(v))

C_ell = cp.array([.25,0.5,0.75,1.])

# FIN DES TRUCS A CHANGER


# Function that computes the estimator of the variance in ML & LML :
def estim_var_MLMC(N, d, M, batch_size):
    var_results = []
    for i in range(0, N, batch_size):
        actual_batch_size = min(batch_size, N - i)
        # Variance at level 0
        X_M_ell = cp.random.standard_normal((actual_batch_size,d,M[0]))
        Y_ell_M_ell = f_vec(X_M_ell,v_0)
        Var_MLMC_batch = cp.var(Y_ell_M_ell, axis=1, ddof=1)
    
        # Loop on M to have correction V^(ell)_{M_ell}(Y_ell) - V^(ell)_{M_ell}(Y_{ell-1})
        for ell in range(1,len(M)):
            X_M_ell = cp.random.standard_normal((actual_batch_size,d,M[ell]))
            Y_ell_M_ell = f_vec(X_M_ell,v_array[ell])
            Y_ell_minus_1_M_ell = f_vec(X_M_ell,v_array[ell-1])
            Var_MLMC_batch += cp.var(Y_ell_M_ell, axis=1, ddof=1) - cp.var(Y_ell_minus_1_M_ell, axis=1, ddof=1)

        var_results.extend(Var_MLMC_batch)

        # Free memory
        X_M_ell = None
        cp.get_default_memory_pool().free_all_blocks()

    return cp.array(var_results) # return var_results (len = N)

def estim_var_MLMC_log(N, d, M, batch_size):
    var_results = []
    for i in range(0, N, batch_size):
        actual_batch_size = min(batch_size, N - i)
        # Variance at level 0
        X_M_ell = cp.random.standard_normal((actual_batch_size,d,M[0]))
        Y_ell_M_ell = f_vec(X_M_ell,v_0)
        log_Var_log_MLMC = cp.log(cp.var(Y_ell_M_ell, axis=1, ddof=1))
    
        # Loop on M to have correction V^(ell)_{M_ell}(Y_ell) - V^(ell)_{M_ell}(Y_{ell-1})
        for ell in range(1,len(M)):
            X_M_ell = cp.random.standard_normal((actual_batch_size,d,M[ell]))
            Y_ell_M_ell = f_vec(X_M_ell,v_array[ell])
            Y_ell_minus_1_M_ell = f_vec(X_M_ell,v_array[ell-1])
            log_Var_log_MLMC += np.log(cp.var(Y_ell_M_ell, axis=1, ddof=1)) - cp.log(cp.var(Y_ell_minus_1_M_ell, axis=1, ddof=1))

        Var_MLMC_log_batch = cp.exp(log_Var_log_MLMC)
    
        var_results.extend(Var_MLMC_log_batch) 

        # Free memory if necessary
        X_M_ell = None
        cp.get_default_memory_pool().free_all_blocks()

    return cp.array(var_results) # return var_results (len = N)


# Function that computes the sample size of each level for a given budget eta
def M_ell_tab(eta, V_ell, C_ell,m=0):
    """
    eta : budget
    V_ell : tableau contenant [V_0,V_1,...,V_L]
    C_ell : tableau contenant [C_0,C_1,...,C_L]

    return : la formule M_ell au dessus qui nous donne le nombre d'élement optimal à simuler
    """
    eta_tilde = eta # - m*cp.sum(...)
    S_L = cp.sum(cp.sqrt(cp.multiply(V_ell,C_ell)))

    return m + 1 + cp.floor(
        (eta_tilde/S_L)*cp.sqrt(V_ell*1/C_ell)
    )


# Computation of our V_ell with N_pilote samples
N_pilote = 10000
X = cp.random.standard_normal((d,N_pilote))
V = [cp.var(f_vec_squared(X,v_0))]
for i in range(1,len(v_array)):
    V.extend([cp.var(f_vec_squared(X,v_array[i]) - f_vec_squared(X,v_array[i-1]))])

V_ell = cp.array(V)

bias_tab = []
var_MLMC_tab = []
bias_log_tab = []
var_MLMC_log_tab = []


# Since we can't compute all at once due to M_0 and M_1 increasing, we need to separate the caluclations by computing batch_size first
def get_max_batch_size(d, M, free_memory, memory_utilization=0.8):
    """
    Calculate the maximum batch size directly based on the memory usage per data sample and the available memory.
    """
    # Calculate the memory usage per sample
    bytes_per_number = cp.dtype('float32').itemsize
    omega = (d+1) * bytes_per_number * (np.sum(M))  # Total memory used per sample

    # Calculate the maximum amount of memory available for batches
    max_memory_per_batch = free_memory * memory_utilization

    # Calculate the maximum batch size that can fit within the memory limit
    batch_size = int(max_memory_per_batch / omega)  # Use int to ensure we get a whole number

    return batch_size


# Get the current free memory on CUDA device
free_memory = cp.cuda.Device().mem_info[0]

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
        M = list(map(int, M_ell_tab(eta, V_ell, C_ell)))

        # Compute the batch size
        batch_size = get_max_batch_size(d, M, free_memory)
        if batch_size == 0:
            break  # Exit if batch size is too small

        # Calculate estimators for MLMC
        Tab_var_MLMC = estim_var_MLMC(N, d, M, batch_size)
        var_MLMC = cp.var(Tab_var_MLMC)
        bias_MLMC = cp.square(cp.mean(Tab_var_MLMC) - exp_var)
        var_MLMC_tab.append(var_MLMC)
        bias_tab.append(bias_MLMC)

        Tab_var_MLMC = None
        cp.get_default_memory_pool().free_all_blocks()

        # Calculate estimators for MLMC log
        Tab_var_MLMC_log = estim_var_MLMC_log(N, d, M, batch_size)
        var_MLMC_log = cp.var(Tab_var_MLMC_log)
        bias_MLMC_log = cp.square(cp.mean(Tab_var_MLMC_log) - exp_var)
        var_MLMC_log_tab.append(var_MLMC_log)
        bias_log_tab.append(bias_MLMC_log)

        Tab_var_MLMC_log = None
        cp.get_default_memory_pool().free_all_blocks()

        # Append results to list and write to file
        results.append({
            "eta": eta,
            "M": M,
            "batch_size": batch_size,
            "var_MLMC": var_MLMC.item(),
            "bias_MLMC": bias_MLMC.item(),
            "var_MLMC_log": var_MLMC_log.item(),
            "bias_MLMC_log": bias_MLMC_log.item()
        })
        file.write(json.dumps(results[-1]) + "\n")  # Write the latest result as a new line in JSON format
        file.flush()

# After the loop, generate the new filename with datetime
new_filename = f'computation_progress_{date_time}.json'

# Rename the file
os.rename(temp_filename, new_filename)

# 
var_MLMC_tab = [x.item() for x in var_MLMC_tab]
bias_tab = [x.item() for x in bias_tab]

var_MLMC_log_tab = [x.item() for x in var_MLMC_log_tab]
bias_log_tab = [x.item() for x in bias_log_tab]


# Plot log-log
plt.figure(figsize=(10, 6))

# Plot des données
plt.loglog(budget_eta, bias_tab, marker='o', label='Biais^2 MLMC')
# plt.loglog(budget_eta, var_MLMC_tab, marker='s', label='Var MLMC Tab')
plt.loglog(budget_eta, bias_log_tab, marker='^', label='Biais^2 Log MLMC', c='g')
# plt.loglog(budget_eta, var_MLMC_log_tab, marker='x', label='Var MLMC Log Tab')

# Ajouter titre et légendes
plt.title(f'Biais^2 en fonction du budget pour n={N}')
plt.xlabel('Budget eta')
plt.ylabel('Bias^2')
plt.legend()


# Configurer la grille
plt.grid(True)

# Afficher le plot
#plt.show()

# Sauvegarder le plot avec la date et l'heure dans le nom du fichier
plt.savefig(f'bias_figsave/graph_{date_time}.png')


# Fermer la figure pour libérer la mémoire
plt.close()