import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from getdist import loadMCSamples, plots
samples = loadMCSamples('chains/living_toe_final')
g = plots.get_single_plotter(width_inch=6, ratio=1)
g.plot_2d(samples, 'w', 'wa', filled=True, colors=['#3b82f6', '#93c5fd'])
ax = g.get_axes()
ax.plot(-1.0, 0.0, marker='+', color='red', markersize=15, markeredgewidth=2, zorder=10)
ax.annotate(r'$\Lambda$CDM', (-1.0, 0.0), textcoords="offset points", xytext=(12, 8), fontsize=11, color='red', fontweight='bold')
ax.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
ax.axvline(-1, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
ax.set_title('Living TOE MCMC: $w_0$ vs $w_a$ (68%/95% CL)', fontsize=12)
g.export('figure1_w_vs_wa.png', dpi=300)
print("Saved figure1_w_vs_wa.png")
