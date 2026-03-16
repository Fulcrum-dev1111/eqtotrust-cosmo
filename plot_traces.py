import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
data = np.loadtxt('chains/living_toe_final.1.txt')
params = ['w', 'wa', r'$\Omega_b h^2$', r'$\Omega_c h^2$', '$H_0$', r'$\tau$', r'$\ln(10^{10}A_s)$', '$n_s$']
cols = [8, 9, 2, 3, 4, 5, 6, 7]
fig, axes = plt.subplots(4, 2, figsize=(14, 16))
fig.suptitle('MCMC Trace Plots - Living TOE Final (53,120 samples)', fontsize=14, y=0.98)
for i, (col, label) in enumerate(zip(cols, params)):
    ax = axes[i // 2, i % 2]
    ax.plot(data[:, col], linewidth=0.3, alpha=0.7, color='#3b82f6')
    ax.set_ylabel(label, fontsize=11)
    ax.set_xlabel('Step' if i >= 6 else '', fontsize=10)
    mean = np.mean(data[:, col])
    ax.axhline(mean, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.tick_params(labelsize=8)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('supplementary_traces.png', dpi=200, bbox_inches='tight')
print("Saved supplementary_traces.png")
