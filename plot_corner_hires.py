import matplotlib
matplotlib.use('Agg')
from getdist import loadMCSamples, plots
samples = loadMCSamples('chains/living_toe_final')
g = plots.get_subplot_plotter(width_inch=10)
g.settings.axes_fontsize = 10
g.settings.lab_fontsize = 12
g.settings.legend_fontsize = 11
g.triangle_plot(samples, ['w', 'wa', 'omegabh2', 'omegach2', 'H0', 'ns'], filled=True, title_limit=1)
g.export('figure2_corner_full.png', dpi=300)
print("Saved figure2_corner_full.png")
