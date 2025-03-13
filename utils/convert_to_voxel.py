import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D


def get_volume_views(volume: torch.Tensor):
    volume = volume.squeeze().__ge__(0.5)
    volume = volume.numpy()
    x, y, z = np.nonzero(volume)
    values = volume[x, y, z]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    sc = ax.scatter(x, y, z, s=10,c=values, cmap='viridis')
    plt.colorbar(sc)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()



def visualize2(volume: torch.tensor):
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))  # Example 3D tensor values
    volume = volume.numpy()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(volume[0],volume[1],volume[2], cmap="coolwarm")

    plt.show()