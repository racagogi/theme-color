import extract as ext
ext.save_colors('./colors/', 'final')
colors = ext.read('./final.h5ad')
colors.plot_contast()
colors.plot_dist()
