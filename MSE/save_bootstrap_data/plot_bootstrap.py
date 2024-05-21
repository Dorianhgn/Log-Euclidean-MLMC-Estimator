import numpy as np
import matplotlib.pyplot as plt

budget_eta = np.load('save_bootstrap_data/budget_eta.npy')
bias_tab_boxplot = np.load('save_bootstrap_data/bootstrap_bias_savefile.npy')
bias_tab_log_boxplot = np.load('save_bootstrap_data/bootstrap_bias_log_savefile.npy')

# Boxplot
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12, 6))


ax1.boxplot(bias_tab_boxplot,positions=budget_eta, widths=(budget_eta*4)/budget_eta[0])
ax2.boxplot(bias_tab_log_boxplot, positions=budget_eta, widths=(budget_eta*4)/budget_eta[0])

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim([1e-11,5e-3])

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylim([1e-11,5e-3])

fig.tight_layout()

plt.show()

print(budget_eta)