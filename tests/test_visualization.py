"""Tests for visualization module."""

import numpy as np
import pytest

# Skip all tests if matplotlib is not available
pytest.importorskip("matplotlib")

from sigdiscovpy.simulation.visualization import (
    plot_ind_curve,
    plot_spatial_cells,
)


class TestPlotSpatialCells:
    """Tests for plot_spatial_cells function."""

    def test_basic_plot(self):
        """Test basic spatial cell plot."""
        import matplotlib.pyplot as plt

        coords = np.random.randn(100, 2) * 100
        ax = plot_spatial_cells(coords, title="Test Plot")

        assert ax is not None
        plt.close()

    def test_with_cell_types(self):
        """Test plot with cell types."""
        import matplotlib.pyplot as plt

        coords = np.random.randn(100, 2) * 100
        cell_types = np.array(["A"] * 50 + ["B"] * 50)

        ax = plot_spatial_cells(coords, cell_types=cell_types)

        assert ax is not None
        plt.close()

    def test_with_expression(self):
        """Test plot with expression coloring."""
        import matplotlib.pyplot as plt

        coords = np.random.randn(100, 2) * 100
        expression = np.random.randn(100)

        ax = plot_spatial_cells(coords, expression=expression, cmap="viridis")

        assert ax is not None
        plt.close()

    def test_highlight_cells(self):
        """Test highlighting sender/receiver cells."""
        import matplotlib.pyplot as plt

        coords = np.random.randn(100, 2) * 100
        sender_indices = np.array([0, 1, 2, 3, 4])
        receiver_indices = np.array([10, 11, 12, 13, 14])

        ax = plot_spatial_cells(
            coords, sender_indices=sender_indices, receiver_indices=receiver_indices
        )

        assert ax is not None
        plt.close()


class TestPlotINDCurve:
    """Tests for plot_ind_curve function."""

    def test_basic_curve(self):
        """Test basic I_ND curve plot."""
        import matplotlib.pyplot as plt

        radii = [50, 100, 150, 200, 250]
        ind_values = [0.1, 0.3, 0.5, 0.4, 0.2]

        ax = plot_ind_curve(radii, ind_values, title="Test I_ND Curve")

        assert ax is not None
        plt.close()

    def test_with_lambda(self):
        """Test curve with lambda estimate."""
        import matplotlib.pyplot as plt

        radii = [50, 100, 150, 200, 250]
        ind_values = [0.1, 0.3, 0.5, 0.4, 0.2]

        ax = plot_ind_curve(radii, ind_values, lambda_est=150)

        assert ax is not None
        plt.close()
