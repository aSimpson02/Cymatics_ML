import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  

import matplotlib.pyplot as plt


def freq_to_modes(freq_hz: float):
    """
    Map a frequency in Hz to a set of integer mode numbers.

    Instead of crude buckets we derive modes from the frequency so that
    small changes in freq produce different (but related) patterns.
    """
    if freq_hz is None or freq_hz <= 0:
        base = 10
    else:
        base = int(round(freq_hz * 10))


    n1 = 2 + (base % 7)          
    n2 = 3 + ((base // 2) % 7)   
    n3 = 4 + ((base // 3) % 7)   


    modes = sorted({n1, n2, n3})
    if len(modes) == 1:
        modes = [modes[0], modes[0] + 1, modes[0] + 2]

    return tuple(modes)


def generate_cymatic_pattern(size=600, modes=(3, 5, 7)):
    """
    Generate a 2D standing-wave interference pattern from given modes.

    We mix several sin*sin combinations with phase shifts and a non-linearity
    so the shapes look richer and more distinct between frequency sets.
    """
    x = np.linspace(-np.pi, np.pi, size)
    y = np.linspace(-np.pi, np.pi, size)
    X, Y = np.meshgrid(x, y)

    m1, m2, m3 = modes

    pattern = (
        np.sin(m1 * X) * np.sin(m2 * Y)
        + np.sin(m2 * X + np.pi / 4) * np.sin(m3 * Y)
        + np.sin(m3 * X + np.pi / 2) * np.sin(m1 * Y)
    )


    pattern /= np.max(np.abs(pattern))
    pattern = np.tanh(1.4 * pattern)

    return pattern


def save_cymatic_image(freq_hz: float, out_path: str):
    """
    Given a frequency, generate & save a cymatic-style PNG.
    """
    modes = freq_to_modes(freq_hz)
    print(f"[cymatics] freq={freq_hz:.3f} Hz -> modes={modes}")

    pattern = generate_cymatic_pattern(size=600, modes=modes)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.imshow(pattern, extent=[-1, 1, -1, 1])
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

    return str(out_path)
