from pathlib import Path

from PyEMD import Visualisation, CEEMDAN
from matplotlib import pyplot as plt


def plot_imfs_from_obj(ceemdan: CEEMDAN, title: str, filename: str | Path):
    """Plot component imfs for a CEEMDAN object, and save to file"""
    vis = Visualisation(ceemdan)
    vis.plot_imfs()

    fig = plt.gcf()
    ax = fig.axes[0]
    ax.set_title(title)

    plt.savefig(filename)
    plt.close()


def plot_imfs(imfs, title: str, filename: str | Path):
    """Plot component imfs without residue, and save to file"""
    vis = Visualisation()
    vis.plot_imfs(imfs=imfs, include_residue=False)

    fig = plt.gcf()
    ax = fig.axes[0]
    ax.set_title(title)

    plt.savefig(filename)
    plt.close()
