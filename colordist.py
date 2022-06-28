import numpy as np
import seaborn as sns
import skimage as si
import matplotlib.pyplot as plt

h = ["#242424",
     "#636363",
     "#989898",
     "#d1d1d1",
     "#F9F9F9",
     "#A34451",
     "#905427",
     "#00734D",
     "#006EA3",
     "#884d8b",
     "#ff85a8",
     "#FFAF43",
     "#FFCA0F",
     "#3eef73",
     "#00F1FF",
     "#a4c6ff", ]
h = list(map(lambda s: s.lstrip('#'), h))
h = list(map(lambda s: list((int(s[i:i+2], 16)/255) for i in (0, 2, 4)), h))

def rgb(rgb1, rgb2):
    for r, g, b in (rgb1, rgb2):
        if not 0.0 <= r <= 1.0:
            raise ValueError("r is out of valid range (0.0 - 1.0)")
        if not 0.0 <= g <= 1.0:
            raise ValueError("g is out of valid range (0.0 - 1.0)")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b is out of valid range (0.0 - 1.0)")

    l1 = _relative_luminance(*rgb1)
    l2 = _relative_luminance(*rgb2)

    if l1 > l2:
        return (l1 + 0.05) / (l2 + 0.05)
    else:
        return (l2 + 0.05) / (l1 + 0.05)


def _relative_luminance(r, g, b):
    r = _linearize(r)
    g = _linearize(g)
    b = _linearize(b)

    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _linearize(v):
    if v <= 0.03928:
        return v / 12.92
    else:
        return ((v + 0.055) / 1.055) ** 2.4


def passes_AA(contrast, large=False):
    if large:
        return contrast >= 3.0
    else:
        return contrast >= 4.5


def passes_AAA(contrast, large=False):
    if large:
        return contrast >= 4.5
    else:
        return contrast >= 7.0


def plot_dist(l):
    collist = l
    coldist = []
    for i in collist:
        temp = []
        for j in collist:
            temp.append(si.color.deltaE_ciede2000(i, j))
        coldist.append(temp)
    mask = np.zeros_like(coldist)
    mask[np.triu_indices_from(mask)] = True
    ax = sns.heatmap(np.array(coldist), mask=mask, linewidth=0.5)
    plt.show()


plot_dist(h)
