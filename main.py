import extract as ext
ext.save_colors('./name/', 'names')
colors = ext.read('./names.h5ad')
colors.plot_hsv()
colors.plot_lab()
colors.plot_rgb()
colors.plot_yiq()
colors.plot_yuv()
colors.ploting_stat()
