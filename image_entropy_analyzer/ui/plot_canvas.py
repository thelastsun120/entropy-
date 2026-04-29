from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width: float = 5, height: float = 3, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def clear(self) -> None:
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

    def plot_histogram(self, hist: np.ndarray) -> None:
        self.clear()
        self.ax.plot(np.arange(256), hist, color="#1f77b4")
        self.ax.set_title("Gray Histogram")
        self.ax.set_xlabel("Gray Level")
        self.ax.set_ylabel("Pixel Count")
        self.ax.grid(alpha=0.2)
        self.draw()

    def plot_entropy_map(self, entropy_map: np.ndarray) -> None:
        self.clear()
        im = self.ax.imshow(entropy_map, cmap="hot")
        self.ax.set_title("Local Entropy Heatmap")
        self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
        self.draw()

    def plot_entropy_comparison(self, original_entropy: float, noisy_entropy: float) -> None:
        self.clear()
        self.ax.bar(["Original", "Noisy"], [original_entropy, noisy_entropy], color=["#2ca02c", "#d62728"])
        self.ax.set_ylabel("Entropy / bit")
        self.ax.set_title("Entropy Comparison")
        self.draw()
