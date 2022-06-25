import numpy as np
import skimage.color as col
import re
import anndata as ad
import os
import pathlib
import matplotlib.pyplot as plt
import pandas as pd

rgb16 = re.compile(r'(?:[0-9a-fA-F]){6}')
theme = re.compile(r'\d+$')
var_names = ["hsv_h", "hsv_s", "hsv_v",
             "yiq_y", "yiq_i", "yiq_q",
             "yuv_y", "yuv_u", "yuv_v",
             "lab_l", "lab_a", "lab_b",
             "rgb_r", "rgb_g", "rgb_b"]


class colors(ad.AnnData):
    def __init__(self, X, obs, dtype):
        super().__init__(X, dtype=dtype)
        self.var_names = var_names
        self.obs_names = obs
        color_df = self.to_df().to_numpy()
        mean = {}
        std = {}
        for i in range(self.shape[1]):
            mean[var_names[i]] = color_df[:, i].mean().round(2)
            std[var_names[i]] = color_df[:, i].std().round(2)
        self.uns['mean'] = mean
        self.uns['std'] = std
        self.obs['theme'] = pd.Categorical(
            list(map(lambda s: theme.sub('', s), self.obs.index)))

    def ploting_stat(self):
        mean_list = self.uns['mean'].items()
        std_list = self.uns['std'].items()
        x, y = zip(*mean_list)
        fig = plt.figure()
        ax = fig.add_subplot(121,)
        ax.bar(x, y, width=0.4)
        x, y = zip(*std_list)
        ax = fig.add_subplot(122,)
        ax.bar(x, y, width=0.4)
        plt.title('stat')
        plt.savefig('stat.png')

    def plot_hsv(self):
        color_df = self.to_df().to_numpy()
        x, y, z = np.hsplit(color_df[:, 0:3], 3)
        axistitle = var_names[0:3]
        colors = color_df[:, 12:15]
        return plot_space(x, y, z, axistitle, colors)

    def plot_yiq(self):
        color_df = self.to_df().to_numpy()
        x, y, z = np.hsplit(color_df[:, 3:6], 3)
        axistitle = var_names[3:6]
        colors = color_df[:, 12:15]
        return plot_space(x, y, z, axistitle, colors)

    def plot_yuv(self):
        color_df = self.to_df().to_numpy()
        x, y, z = np.hsplit(color_df[:, 6:9], 3)
        axistitle = var_names[6:9]
        colors = color_df[:, 12:15]
        return plot_space(x, y, z, axistitle, colors)

    def plot_lab(self):
        color_df = self.to_df().to_numpy()
        x, y, z = np.hsplit(color_df[:, 9:12], 3)
        axistitle = var_names[9:12]
        colors = color_df[:, 12:15]
        return plot_space(x, y, z, axistitle, colors, )

    def plot_rgb(self):
        color_df = self.to_df().to_numpy()
        x, y, z = np.hsplit(color_df[:, 12:15], 3)
        axistitle = var_names[12:15]
        colors = color_df[:, 12:15]
        return plot_space(x, y, z, axistitle, colors)
        # a[a.obs['theme']== a.obs['theme'].cat.categories[2],:]


def plot_space(x, y, z, axistitle, colors):
    fig = plt.figure()
    ax = fig.add_subplot(221,)
    ax.scatter(x=x, y=y, s=10,  c=colors,)
    ax.set_xlabel(axistitle[0])
    ax.set_ylabel(axistitle[1])
    ax = fig.add_subplot(222,)
    ax.scatter(x=y, y=z, s=10,  c=colors,)
    ax.set_xlabel(axistitle[1])
    ax.set_ylabel(axistitle[2])
    ax = fig.add_subplot(223,)
    ax.scatter(x=z, y=x, s=10,  c=colors,)
    ax.set_xlabel(axistitle[2])
    ax.set_ylabel(axistitle[0])
    plt.show()


def read(filename):
    adata = ad.read(filename)
    return colors(adata.X, adata.obs_names, dtype='float32')


def flat(lst):
    return [x for y in lst for x in y]


def hex_rgb(hexrgb):
    r, g, b = (int(hexrgb[i*2:i*2+2], 16)/255 for i in range(3))
    return np.array([r, g, b])


def colstat(color):
    hsv = col.rgb2hsv(color)
    yiq = col.rgb2yiq(color)
    yuv = col.rgb2yuv(color)
    lab = col.rgb2lab(color)
    return np.array((hsv, yiq, yuv, lab, color)).round(2).flatten()


def file_extract(filename):
    theme_name = str(filename).split('/')[-1].split('.')[0]
    with open(filename, 'r', errors='ignore') as f:
        colors = f.readlines()
        colors = list(set(flat(list(map(rgb16.findall, colors)))))
    return (theme_name, colors)


def make_anndata(filename):
    (name, color) = file_extract(filename)
    color = np.array([colstat(hex_rgb(i)) for i in color])
    obs = [name+f'{i:d}' for i in range(color.shape[0])]
    coladata = colors(color, obs, 'float32')
    return coladata


def save_colors(source_name, save_name):
    filenames = [source_name+'/'+i for i in os.listdir(source_name)]
    coladata = ad.concat([make_anndata(i) for i in filenames])
    save_path = pathlib.Path(save_name+".h5ad")
    coladata.write(save_path)
