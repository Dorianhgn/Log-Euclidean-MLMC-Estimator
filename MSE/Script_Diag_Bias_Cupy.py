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

N=10000  # On génère à chaque fois N estimations de nos variances V_ML et V_LML pour calculer leur variance et leur biais
budget_eta = np.logspace(1.5,5,10)  # Budget qui varie entre 10^1 et 10^4 

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
exp_var = cp.sum(cp.square(v_3))
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
MSE_tab = []
var_MLMC_tab = []
bias_log_tab = []
MSE_log_tab = []
var_MLMC_log_tab = []
MSE_log_euclidean_tab = []


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
        # Compute the values we need
        var_MLMC = cp.var(Tab_var_MLMC)
        MSE_MLMC = cp.mean(cp.square(Tab_var_MLMC - exp_var))
        bias_MLMC = cp.abs(cp.mean(Tab_var_MLMC) - exp_var)
        # Add them in a array
        var_MLMC_tab.append(var_MLMC)
        MSE_tab.append(MSE_MLMC)
        bias_tab.append(bias_MLMC)

        Tab_var_MLMC = None
        cp.get_default_memory_pool().free_all_blocks()

        # Calculate estimators for MLMC log
        Tab_var_MLMC_log = estim_var_MLMC_log(N, d, M, batch_size)
        # Compute the values we need
        var_MLMC_log = cp.var(Tab_var_MLMC_log)
        MSE_MLMC_log = cp.mean(cp.square(Tab_var_MLMC_log - exp_var))
        bias_MLMC_log = cp.abs(cp.mean(Tab_var_MLMC_log) - exp_var)

        # Add them in a array
        var_MLMC_log_tab.append(var_MLMC_log)
        MSE_log_tab.append(MSE_MLMC_log)
        bias_log_tab.append(bias_MLMC_log)
        
        #MSE in the log euclidean metric:
        MSE_MLMC_log_euclidean = cp.mean(cp.square(cp.log(Tab_var_MLMC_log) - cp.log(exp_var)))
        MSE_log_euclidean_tab.append(MSE_MLMC_log_euclidean)

        Tab_var_MLMC_log = None
        cp.get_default_memory_pool().free_all_blocks()

        # Append results to list and write to file
        results.append({
            "eta": eta,
            "M": M,
            "batch_size": batch_size,
            "var_MLMC": var_MLMC.item(),
            "bias_MLMC": bias_MLMC.item(),
            "MSE_MLMC": MSE_MLMC.item(),
            "var_MLMC_log": var_MLMC_log.item(),
            "bias_MLMC_log": bias_MLMC_log.item(),
            "MSE_MLMC_log": MSE_MLMC_log.item(),
            "MSE_MLMC_log_euclidean": MSE_MLMC_log_euclidean.item()
        })
        file.write(json.dumps(results[-1]) + "\n")  # Write the latest result as a new line in JSON format
        file.flush()

# After the loop, generate the new filename with datetime
new_filename = f'computation_progress_{date_time}.json'

# Rename the file
os.rename(temp_filename, new_filename)

# getting the tab from cupy so it can be plotted
var_MLMC_tab = [x.item() for x in var_MLMC_tab]
bias_tab = [x.item() for x in bias_tab]
MSE_tab = [x.item() for x in MSE_tab]

var_MLMC_log_tab = [x.item() for x in var_MLMC_log_tab]
bias_log_tab = [x.item() for x in bias_log_tab]
MSE_log_tab = [x.item() for x in MSE_log_tab]

MSE_log_euclidean_tab = [x.item() for x in MSE_log_euclidean_tab]

"""
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
plt.close()"""


# Plot 1: MSE_log_euclidean_tab en fonction de budget_eta
plt.figure(figsize=(10, 6))
plt.loglog(budget_eta, MSE_log_euclidean_tab, marker='o')
plt.xlabel('Budget $\eta$')
plt.ylabel('MSE in the log-Euclidean metric of $\hat{V}^{LML}_4$')
plt.title('MSE in the log-Euclidean metric as a function of budget $\eta$')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# Plot 2: Comparaison de MSE_tab et MSE_log_tab en fonction de budget_eta
plt.figure(figsize=(10, 6))
plt.loglog(budget_eta, MSE_tab, marker='o', label='MSE($\hat{V}^{ML}_4$)')
plt.loglog(budget_eta, MSE_log_tab, marker='x', label='MSE($\hat{V}^{LML}_4$)')
plt.xlabel('Budget $\eta$')
plt.ylabel('MSE')
plt.title('Comparaison of the MSE of estimators in the Euclidean metric as a function of budget $\eta$')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# Plot 3: Comparaison de bias_tab et bias_log_tab avec des barres d'erreur
# Calcul des barres d'erreur
error_bars_tab = np.sqrt(var_MLMC_tab) / np.sqrt(N) 
error_bars_log_tab = np.sqrt(var_MLMC_log_tab) / np.sqrt(N)

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot des biais en log-log
ax1.plot(budget_eta, bias_tab, marker='o', label='bias($\hat{V}^{ML}_4$)', color='blue')
ax1.plot(budget_eta, bias_log_tab, marker='x', label='bias($\hat{V}^{LML}_4$)', color='green')
ax1.set_xlabel('Budget $\eta$')
ax1.set_ylabel('Bias')
ax1.set_title('Comparaison of the bias of estimators as a function of budget $\eta$')
ax1.legend(loc='upper right')
ax1.grid(True, which="both", ls="--")
ax1.set_xscale('log')

# Créer un deuxième axe Y pour les barres d'erreur
ax2 = ax1.twinx()
ax2.set_ylabel('Erreur')
# Plot des barres d'erreur en échelle normale
ax2.errorbar(budget_eta, bias_tab, yerr=error_bars_tab, fmt='x', color='red', alpha=0.5, label='standard deviation of $\hat{V}^{ML}_4$', capsize=5, elinewidth=1)
ax2.errorbar(budget_eta, bias_log_tab, yerr=error_bars_log_tab, fmt='o', color='orange', alpha=0.5, label='standard deviation of $\hat{V}^{LML}_4$', capsize=5, elinewidth=1)

ax2.set_xscale('log')
# Ajuster les limites des axes pour qu'ils soient alignés
ax2.set_ylim(ax1.get_ylim())
ax2.set_xlim(ax1.get_xlim())
ax2.legend(loc='center right')

fig.tight_layout()
plt.show()


# Plot 4: Comparaison des variances de notre estimateur
plt.figure(figsize=(10, 6))

plt.loglog(budget_eta, var_MLMC_tab, marker='s', label='Var MLMC Tab')
plt.loglog(budget_eta, var_MLMC_log_tab, marker='x', label='Var MLMC Log Tab')

plt.title(f'Variance of estimators as a function of budget_eta for n={N}')
plt.xlabel('Budget $\eta$')
plt.ylabel('Var(estimator)')
plt.legend()

plt.grid(True, which="both", ls="--")

plt.show()

# Test
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.errorbar(budget_eta, bias_tab, yerr=(error_bars_tab), fmt='', color='red', alpha=0.5, label='Erreur Bias_tab', capsize=5, elinewidth=1)
ax1.errorbar(budget_eta, bias_log_tab, yerr=(error_bars_log_tab), fmt='', color='orange', alpha=0.5, label='Erreur Bias_log_tab', capsize=5, elinewidth=1)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.grid(True, which="both", ls="--")

plt.show()