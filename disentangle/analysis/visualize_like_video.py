import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from IPython.display import HTML


def dummy_func():
    """
    Thanks to Milli.
    """
    plt.rcParams["animation.html"] = "jshtml"

    # create dummy data
    shape = (64, 64)
    n_frames = 5
    radii = np.linspace(1, shape[0], n_frames)
    images = [
        np.zeros(shape, dtype=np.uint8)
        for _ in range(n_frames)
    ]
    ii, jj = np.mgrid[:shape[0], :shape[1]]
    for r, image in zip(radii, images):
        image[(ii-shape[0]//2) ** 2 + (jj-shape[1]//2) ** 2 <= r ** 2] = 255

    # --- animation code
    fig, ax = plt.subplots()
    im = ax.imshow(images[0])
    # update function that defines the matplotlib animation
    def update(frame):
        im.set_array(images[frame])
        return [im]

    anim = FuncAnimation(
        fig, update, frames=n_frames, interval=200
    )
    HTML(anim.to_jshtml())
