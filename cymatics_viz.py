import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  

import matplotlib.pyplot as plt


def freq_to_modes(freq_hz: float):
    """
    Map a frequency in Hz to some simple mode numbers.
    This is heuristic – tweak as you like.
    """
    if freq_hz < 1.0:
        return (2, 3, 5)
    elif freq_hz < 2.0:
        return (3, 5, 7)
    elif freq_hz < 3.0:
        return (4, 6, 8)
    else:
        return (5, 7, 9)


def generate_cymatic_pattern(size=500, modes=(2, 3, 5)):
    """
    Make a 2D standing-wave interference pattern.
    """
    x = np.linspace(-np.pi, np.pi, size)
    y = np.linspace(-np.pi, np.pi, size)
    X, Y = np.meshgrid(x, y)

    pattern = np.zeros_like(X)
    for n in modes:
        pattern += np.sin(n * X) * np.sin(n * Y)

    pattern = pattern / np.max(np.abs(pattern))
    return pattern


def save_cymatic_image(freq_hz: float, out_path: str):
    """
    Given a frequency, generate & save a cymatic-style PNG.
    """
    modes = freq_to_modes(freq_hz)
    pattern = generate_cymatic_pattern(size=500, modes=modes)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.imshow(pattern, extent=[-1, 1, -1, 1])
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

    return str(out_path)
